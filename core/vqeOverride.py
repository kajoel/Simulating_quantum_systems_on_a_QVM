"""
@author: Sebastian (fast självklart 99% rigetti, jag bara modifierade den)
"""
"""
Updated vqe_expectation och expectation from sampling,
@author Eric utöver Rigetti

Probably need some work!!! Not complete
"""

from grove.pyvqe.vqe import VQE, OptResults
from collections import Counter
from typing import List, Union

import funcsigs
import numpy as np
from pyquil import Program
from pyquil.api import QuantumComputer, WavefunctionSimulator
from pyquil.api._qvm import QVM
from pyquil.gates import RX, RY, MEASURE, STANDARD_GATES
from pyquil.paulis import PauliTerm, PauliSum


class VQE_override(VQE):

    def __init__(self, minimizer, minimizer_args=[], minimizer_kwargs={}):
        super().__init__(minimizer, minimizer_args, minimizer_kwargs)

    def vqe_run(self, variational_state_evolve, hamiltonian, initial_params,
                gate_noise=None, measurement_noise=None,
                jacobian=None, qc=None, disp=False, samples=None,
                return_all=False):
        """
        functional minimization loop.

        :param variational_state_evolve: function that takes a set of parameters
                                        and returns a pyQuil program.
        :param hamiltonian: (PauliSum) object representing the hamiltonian of
                            which to take the expectation value.
        :param initial_params: (ndarray) vector of initial parameters for the
                               optimization
        :param gate_noise: list of Px, Py, Pz probabilities of gate being
                           applied to every gate after each get application
        :param measurement_noise: list of Px', Py', Pz' probabilities of a X, Y
                                  or Z being applied before a measurement.
        :param jacobian: (optional) method of generating jacobian for parameters
                         (Default=None).
        :param qc: (optional) QuantumComputer object.
        :param disp: (optional, bool/callable) display level. If True then
                each iteration
                expectation and parameters are printed at each optimization
                iteration. If callable, called after each iteration. The signature:

                    ``disp(xk, state)``

                where ``xk`` is the current parameter vector. and ``state``
                is the current expectation value.

        :param samples: (int) Number of samples for calculating the expectation
                        value of the operators.  If `None` then faster method
                        ,dotting the wave function with the operator, is used.
                        Default=None.
        :param return_all: (optional, bool) request to return all intermediate
                           parameters determined during the optimization.
        :return: (vqe.OptResult()) object :func:`OptResult <vqe.OptResult>`.
                 The following fields are initialized in OptResult:
                 -x: set of w.f. ansatz parameters
                 -fun: scalar value of the objective function

                 -iteration_params: a list of all intermediate parameter vectors. Only
                                    returned if 'return_all=True' is set as a vqe_run()
                                    option.

                 -expectation_vals: a list of all intermediate expectation values. Only
                                    returned if 'return_all=True' is set as a
                                    vqe_run() option.
        """

        """
        if (disp is not None and disp is not True and
                disp is not False):
            self._disp_fun = disp
        else: pass
        """

        self._disp_fun = print

        iteration_params = []
        expectation_vals = []
        expectation_vars = []
        self._current_expectation = None
        self._current_variance = None

        if qc is None:
            qubits = hamiltonian.get_qubits()
            qc = QuantumComputer(name=f"{len(qubits)}q-noisy-qvm",
                                 qam=QVM(gate_noise=gate_noise,
                                         measurement_noise=measurement_noise))
        else:
            self.qc = qc

        def objective_function(params):

            """
            closure representing the functional

            :param params: (ndarray) vector of parameters for generating the
                           the function of the functional.
            :return: (float) expectation value
            """
            pyquil_prog = variational_state_evolve(params)
            mean_value, tmp_vars = self.expectation(pyquil_prog,
                                                    hamiltonian, samples,
                                                    qc)
            self._current_variance = tmp_vars
            self._current_expectation = mean_value  # store for printing

            return mean_value

        def print_current_iter(iter_vars):
            self._disp_fun("\tParameters: {} ".format(iter_vars))
            if jacobian is not None:
                grad = jacobian(iter_vars)
                self._disp_fun(
                    "\tGrad-L1-Norm: {}".format(np.max(np.abs(grad))))
                self._disp_fun(
                    "\tGrad-L2-Norm: {} ".format(np.linalg.norm(grad)))

            self._disp_fun("\tE => {}".format(self._current_expectation))

            if return_all:
                iteration_params.append(iter_vars)
                expectation_vals.append(self._current_expectation)
                expectation_vars.append(self._current_variance)

        # using self.minimizer
        arguments = funcsigs.signature(self.minimizer).parameters.keys()

        def callback(iter_vars):
            disp(iter_vars, self._current_expectation, self._current_variance)
            if return_all:
                iteration_params.append(iter_vars)
                expectation_vals.append(self._current_expectation)
                expectation_vars.append(self._current_variance)

        if disp is True and 'callback' in arguments:
            self.minimizer_kwargs['callback'] = print_current_iter

        if (callable(disp)) and 'callback' in arguments:
            self.minimizer_kwargs['callback'] = callback

        args = [objective_function, initial_params]
        args.extend(self.minimizer_args)
        if 'jac' in arguments:
            self.minimizer_kwargs['jac'] = jacobian

        result = self.minimizer(*args, **self.minimizer_kwargs)

        if hasattr(result, 'status'):
            if result.status != 0:
                self._disp_fun(
                    "Classical optimization exited with an error index: %i"
                    % result.status)

        results = OptResults()
        if hasattr(result, 'x'):
            results.x = result.x
            results.fun = result.fun
        else:
            results.x = result

        if return_all:
            # iteration_params.append(result['x'][0])
            # expectation_vals.append(result['fun'])
            results.iteration_params = iteration_params
            results.expectation_vals = expectation_vals
            results.expectation_vars = expectation_vars
        return results

    @staticmethod
    def expectation(pyquil_prog: Program,
                    pauli_sum: Union[PauliSum, PauliTerm, np.ndarray],
                    samples: int,
                    qc: QuantumComputer) -> tuple:
        """
        Compute the expectation value of pauli_sum over the distribution
        generated from pyquil_prog. Updated by Eric


        :param pyquil_prog: The state preparation Program to calculate the expectation value of.
        :param pauli_sum: PauliSum representing the operator of which to calculate the expectation
            value or a numpy matrix representing the Hamiltonian tensored up to the appropriate
            size.
        :param samples: The number of samples used to calculate the expectation value. If samples
            is None then the expectation value is calculated by calculating <psi|O|psi>. Error
            models will not work if samples is None.
        :param qc: The QuantumComputer object.

        :return: A float representing the expectation value of pauli_sum given the distribution
            generated from quil_prog.

        @author: Eric
        """
        if isinstance(pauli_sum, np.ndarray):
            # debug mode by passing an array
            wf = WavefunctionSimulator().wavefunction(pyquil_prog)
            wf = np.reshape(wf.amplitudes, (-1, 1))
            average_exp = np.conj(wf).T.dot(pauli_sum.dot(wf)).real
            print('Variance 0 for debug properties')
            return average_exp, 0.0
        else:
            if not isinstance(pauli_sum, (PauliTerm, PauliSum)):
                raise TypeError(
                    "pauli_sum variable must be a PauliTerm or PauliSum object")

            if isinstance(pauli_sum, PauliTerm):
                pauli_sum = PauliSum([pauli_sum])

            if samples is None:
                '''
                operator_progs = []
                operator_coeffs = []
                for p_term in pauli_sum.terms:
                    op_prog = Program()
                    for qindex, op in p_term:
                        op_prog.inst(STANDARD_GATES[op](qindex))
                    operator_progs.append(op_prog)
                    operator_coeffs.append(p_term.coefficient)

                result_overlaps = WavefunctionSimulator().expectation(pyquil_prog, pauli_sum.terms)
                result_overlaps = list(result_overlaps)
                assert len(result_overlaps) == len(operator_progs),\
                    """Somehow we didn't get the correct number of results back from the QVM"""
                expectation = sum(list(map(lambda x: x[0] * x[1],
                                           zip(result_overlaps, operator_coeffs))))
                '''
                result_overlaps = WavefunctionSimulator().expectation(
                    pyquil_prog, pauli_sum.terms)
                expectation = np.sum(result_overlaps)
                return expectation.real, 0.0
            else:
                if not isinstance(samples, int):
                    raise TypeError("samples variable must be an integer")
                if samples <= 0:
                    raise ValueError(
                        "samples variable must be a positive integer")

                # normal execution via fake sampling
                # stores the sum of contributions to the energy from
                # each operator term
                expectation = 0.0
                variance = 0.0
                term_sum = sum(
                    np.abs(term.coefficient) for term in pauli_sum.terms)
                for j, term in enumerate(pauli_sum.terms):
                    meas_basis_change = Program()
                    qubits_to_measure = []
                    if len(term.operations_as_set()) == 0:
                        meas_outcome = 1.0
                        meas_vars = 0.0
                    else:
                        for index, gate in term:
                            qubits_to_measure.append(index)
                            if gate == 'X':
                                meas_basis_change.inst(RY(-np.pi / 2, index))
                            elif gate == 'Y':
                                meas_basis_change.inst(RX(np.pi / 2, index))

                        meas_outcome, meas_vars = \
                            expectation_from_sampling(
                                pyquil_prog + meas_basis_change,
                                qubits_to_measure,
                                qc,
                                int(round(samples * np.abs(
                                    term.coefficient) / term_sum)))

                    expectation += term.coefficient * meas_outcome
                    variance += (np.abs(term.coefficient) ** 2) * meas_vars

                return expectation.real, variance.real


def parity_even_p(state, marked_qubits):
    """
    Calculates the parity of elements at indexes in marked_qubits

    Parity is relative to the binary representation of the integer state.

    :param state: The wavefunction index that corresponds to this state.
    :param marked_qubits: The indexes to be considered in the parity sum.
    :returns: A boolean corresponding to the parity.
    """
    assert isinstance(state, int), \
        f"{state} is not an integer. Must call parity_even_p with an integer state."
    mask = 0
    for q in marked_qubits:
        mask |= 1 << q
    return bin(mask & state).count("1") % 2 == 0


def expectation_from_sampling(pyquil_program: Program,
                              marked_qubits: List[int],
                              qc: QuantumComputer,
                              samples: int) -> tuple:
    """
    Calculation of Z_{i} at marked_qubits

    Given a wavefunctions, this calculates the expectation value of the Zi
    operator where i ranges over all the qubits given in marked_qubits.

    :param pyquil_program: pyQuil program generating some state
    :param marked_qubits: The qubits within the support of the Z pauli
                          operator whose expectation value is being calculated
    :param qc: A QuantumComputer object.
    :param samples: Number of bitstrings collected to calculate expectation
                    from sampling.
    :returns: The expectation value and variance as a float.
    """
    program = Program()
    ro = program.declare('ro', 'BIT', max(marked_qubits) + 1)
    program += pyquil_program
    program += [MEASURE(qubit, r) for qubit, r in
                zip(list(range(max(marked_qubits) + 1)), ro)]
    program.wrap_in_numshots_loop(samples)
    executable = qc.compile(program)
    bitstring_samples = qc.run(executable)
    bitstring_tuples = list(map(tuple, bitstring_samples))

    freq = Counter(bitstring_tuples)

    # perform weighted average
    exp_list = []
    for bitstring, count in freq.items():
        bitstring_int = int("".join([str(x) for x in bitstring[::-1]]), 2)
        if parity_even_p(bitstring_int, marked_qubits):
            exp_list.append(count)
        else:
            exp_list.append(-1 * count)
    expectation = np.sum(exp_list) / samples
    variance = (1 - expectation ** 2) / samples
    return expectation, variance
