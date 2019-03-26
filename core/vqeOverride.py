"""
@author: Sebastian (fast självklart 99% rigetti, jag bara modifierade den)
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
        self._current_expectation = None
        if samples is None:
            print("""WARNING: Fast method for expectation will be used. Noise
                     models will be ineffective""")

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
            mean_value = self.expectation(pyquil_prog, hamiltonian, samples, qc)
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

        # using self.minimizer
        arguments = funcsigs.signature(self.minimizer).parameters.keys()

        if disp is True and 'callback' in arguments:
            self.minimizer_kwargs['callback'] = print_current_iter

        if (disp is not None and disp is not True) and 'callback' in arguments:
            self.minimizer_kwargs['callback'] = lambda x: disp(x,
                                                               self._current_expectation)

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
            results.iteration_params = iteration_params
            results.expectation_vals = expectation_vals
        return results
