"""
Created on Mon Mar  4 10:50:49 2019
"""

# Imports
from core import maps
import numpy as np
from pyquil.quil import Program
from grove.alpha.arbitrary_state.arbitrary_state import create_arbitrary_state
from openfermion import FermionOperator, QubitOperator, jordan_wigner, \
    hermitian_conjugated
from forestopenfermion import qubitop_to_pyquilpauli
from pyquil.paulis import PauliSum, PauliTerm, exponential_map, suzuki_trotter
from pyquil.gates import X
from typing import Callable, List, Union
from pyquil import get_qc


def one_particle(h: np.ndarray):
    """
    Creates a function(theta) that creates a program to set up an arbitrary
    one-particle-state.

    @author: Joel, Carl

    :param h: The hamiltonian matrix.
    :return: function(theta) -> pyquil program setting up the state.
    """
    dim = h.shape[0]

    def wrap(theta: np.ndarray) -> Program:
        """
        Creates arbitrary one-particle state.

        :param theta: Coefficients in superposition.
        :return: Pyquil program setting up the state.
        """
        vector = np.zeros(2 ** dim)
        vector[1] = 1 / np.sqrt(dim)
        vector[[2 ** (i + 1) for i in range(dim - 1)]] = theta
        vector *= 1 / np.linalg.norm(vector)
        return create_arbitrary_state(vector)

    return wrap


def multi_particle(h: np.ndarray):
    """
    Creates a function(theta) that creates a program to set up an arbitrary
    state.

    @author: Joel

    :param h: The hamiltonian matrix.
    :return: function(theta) -> pyquil program setting up the state.
    """
    dim = h.shape[0]

    def wrap(theta: np.ndarray):
        """
        Creates arbitrary state.

        :param theta: Vector representing the state.
        :return: PyQuil program setting up the state.
        """
        theta = np.concatenate((np.array([1]) / np.sqrt(dim), theta), axis=0)
        return create_arbitrary_state(theta)

    return wrap


def multi_particle_stereographic(h: np.ndarray):
    """
    Multi particle ansatz using a stereographic projection for dimensionality
    reduction.

    @author = Joel

    :param h: The hamiltonian matrix.
    :return: function(theta) -> pyquil program setting up the state.
    """
    pole = int(np.argmax(np.diag(h)))

    def wrap(theta):
        """
        Creates arbitrary state using stereographic projection.

        :param theta: Vector representing the state.
        :return: PyQuil program setting up the state.
        """
        return create_arbitrary_state(maps.plane_to_sphere(theta, pole))

    return wrap


def one_particle_ucc(h, reference=1, trotter_order=1, trotter_steps=1):
    """
    UCC-style ansatz preserving particle number.

    @author: Joel, Carl

    :param np.ndarray h: The hamiltonian matrix.
    :param int reference: the binary number corresponding to the reference
        state (which must be a Fock-state).
    :param int trotter_order: trotter order in suzuki_trotter
    :param int trotter_steps: trotter steps in suzuki_trotter
    :return: function(theta) which returns the ansatz Program
    """
    dim = h.shape[0]

    terms = []
    for occupied in range(dim):
        if reference & (1 << occupied):
            for unoccupied in range(dim):
                if not reference & (1 << unoccupied):
                    term = FermionOperator(((unoccupied, 1), (occupied, 0))) \
                           - FermionOperator(((occupied, 1), (unoccupied, 0)))
                    terms.append(
                        qubitop_to_pyquilpauli(jordan_wigner(term))
                    )

    exp_maps = trotterize(terms, trotter_order, trotter_steps)

    def wrap(theta):
        """
        Returns the ansatz Program.

        :param np.ndarray theta: parameters
        :return: the Program
        :rtype: pyquil.Program
        """
        prog = Program()
        for qubit in range(int.bit_length(reference)):
            if reference & (1 << qubit):
                prog += X(qubit)
        for idx, exp_map in enumerate(exp_maps):
            for exp in exp_map:
                prog += exp(theta[idx])
        return prog

    return wrap


def multi_particle_ucc(h, reference=0, trotter_order=1, trotter_steps=1):
    """
    UCC-style ansatz that doesn't preserve anything (i.e. uses all basis
    states). This is basically an implementation of create_arbitrary_state
    (however, theta will not match the coefficients in the superposition)
    built on pyquil.paulis.exponentiate_map.

    One idea is to use lowering and raising operators and "check operators"
    instead of Xs. This results in a more straight-forward mapping of theta
    to the coefficients since no previously produced state will be mapped
    to other states in a later stage. To clarify: with the current
    implementation the state |1 1 0> will both be produced by exp(X2 X1)
    operating on |0 0 0> and by exp(X2) operating on exp(X1)|0 0 0> (which
    contains a |0 1 0> term). This could improve potential bad properties
    with the current implementation. However, it might be difficult to
    create commuting hermitian terms, which is required in
    exponential_map_commuting_pauli_terms.

    If this function is called multiple times, particularly if theta has the
    same length in all calls, caching terms might significantly increase
    performance.

    @author: Joel

    :param np.ndarray h: The hamiltonian matrix.
    :param reference: integer representation of reference state
    :param trotter_order: Trotter order in trotterization
    :param trotter_steps: Trotter steps in trotterization
    :return: function(theta) which returns the ansatz Program. -1j*theta[i] is
        the coefficient in front of the term prod_k X_k^bit(i,k) where
        bit(i, k) is the k'th bit of i in binary, in the exponent.
    """
    dim = h.shape[0]

    terms = []
    for state in range(dim):
        if state != reference:
            term = QubitOperator(())
            for qubit in range(int.bit_length(state)):
                if (state ^ reference) & (1 << qubit):
                    # lower/raise qubit
                    term *= QubitOperator((qubit, "X"), 1 / 2) \
                            + QubitOperator((qubit, "Y"), 1j * (int(
                                reference & (1 << qubit) != 0) - 1 / 2))
                else:
                    # check that qubit has correct value (same as i and j)
                    term *= QubitOperator((), 1 / 2) \
                            + QubitOperator((qubit, "Z"), 1 / 2 - int(
                                reference & (1 << qubit) != 0))
            terms.append(
                qubitop_to_pyquilpauli(term - hermitian_conjugated(term))
            )

    exp_maps = trotterize(terms, trotter_order, trotter_steps)

    def wrap(theta):
        """
        Returns the ansatz Program.

        :param np.ndarray theta: parameters
        :return: the Program
        :rtype: pyquil.Program
        """
        prog = Program()
        for qubit in range(int.bit_length(reference)):
            if reference & (1 << qubit):
                prog += X(qubit)
        for idx, exp_map in enumerate(exp_maps):
            for exp in exp_map:
                prog += exp(theta[idx])
        return prog

    return wrap


# ################## ANSATZ RELATED QC:s #######################


def one_particle_qc(h: np.ndarray):
    # TODO: doc
    return get_qc('{}q-qvm'.format(h.shape[0]))


def multi_particle_qc(h: np.ndarray):
    # TODO: doc
    return get_qc('{}q-qvm'.format(int.bit_length(h.shape[0])))


# ################## ANSATZ RELATED FUNCTIONS ##################


def trotterize(terms, trotter_order, trotter_steps) -> List[
        List[Callable[[float], Program]]]:
    """
    Trotterize the terms. If terms = [[t11, t12], [t21, t22]] the
    Trotterization approximates exp(t11+t12)*exp(t21+t22) (not quite correct
    but you get the idea).

    @author = Joel, Carl

    :param List[PauliSum] terms: PauliSums
    :param int trotter_order: trotter order in suzuki_trotter
    :param int trotter_steps: trotter steps in suzuki_trotter
    :return: list of lists of functions(theta) that returns Programs
    """
    # TODO: better docstring
    exp_maps = []
    old_len = None
    order_slices = None
    for term in terms:
        if len(term) != old_len:
            order_slices = suzuki_trotter_karlsson(len(term), trotter_order,
                                                   trotter_steps)
        tmp = []
        for coeff, operator in order_slices:
            tmp.append(exponential_map(-1j * coeff * term[operator]))
        exp_maps.append(tmp)
    return exp_maps


def suzuki_trotter_karlsson(num_op, trotter_order, trotter_steps):
    """
    Generalization of the suzuki_trotter decomposition in the case
    trotter_order=1 to be able to handle exp(A+B+C). To make the approximation
    the following limit is used:
    exp(sum_i A_i) = lim_{n \to inf} [prod_i (1 + exp(A_i)/n) ]^n

    exp(sum_i A_i) is approximated as exp(w1 i1) exp(w2 i2)...

    @author = Joel

    :param int num_op: number of A_i's in exp(sum_i A_i)
    :param int trotter_order: order of decomposition
    :param int trotter_steps: steps of decomposition, n from above
    :return: List of tuples (wk, ik) corresponding to the coefficient and
        operator.
    """

    if num_op == 1:
        order_slices = [(1, 0)]
    elif num_op == 2:
        order_slices = suzuki_trotter(trotter_order, trotter_steps)
    else:
        if trotter_order != 1:
            raise ValueError('suzuki_trotter_karlsson decomposition currently'
                             'requires num_op=2 or trotter_order=1.')
        order_slices = trotter_steps * [(1 / trotter_steps, i)
                                        for i in range(num_op)]
    return order_slices


def exponential_map_commuting_pauli_terms(terms: Union[List[PauliTerm],
                                                       PauliSum]):
    """
    Returns a function f(theta) which, given theta, returns the Program
    corresponding to exp(-1j sum_i theta[i]*term[i]) =
    prod_i exp(-1j*theta[i]*term[i]). Note that the equality only holds if
    the terms are commuting. This was inspired by pyquil.exponential_map
    and pyquil.exponentiate_commuting_pauli_sum.

    @author = Joel

    :param  terms:
        a list of pauli terms
    :return: a function that takes a vector parameter and returns a Program.
    """

    if isinstance(terms, PauliSum):
        terms = terms.terms
    exp_map = []
    for term in terms:
        exp_map.append(exponential_map(term))

    def wrap(theta):
        """
        Returns the ansatz Program.

        :param np.ndarray theta: parameters
        :return: the Program
        :rtype: pyquil.Program
        """
        prog = Program()
        for idx, angle in enumerate(theta):
            prog += exp_map[idx](angle)
        return prog

    return wrap
