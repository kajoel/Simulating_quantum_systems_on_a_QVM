"""
Created on Mon Mar  4 10:50:49 2019
"""

# Imports
import numpy as np
from pyquil.quil import Program
from grove.alpha.arbitrary_state.arbitrary_state import create_arbitrary_state
from openfermion import FermionOperator, QubitOperator, jordan_wigner
from forestopenfermion import qubitop_to_pyquilpauli
from pyquil.paulis import PauliSum, PauliTerm, exponential_map


def one_particle(theta: np.ndarray) -> Program:
    """
    @author: Joel, Carl
    Creates a program to set up an arbitrary one-particle-state.
    :param theta: Vector representing the state.
    :return: PyQuil program setting up the state.
    """

    vector = np.zeros(2 ** (theta.shape[0] + 1))
    vector[1] = 1 / np.sqrt(theta.size + 1)
    vector[[2 ** (i + 1) for i in range(theta.shape[0])]] = theta
    vector *= 1 / np.linalg.norm(vector)
    return create_arbitrary_state(vector)


def multi_particle(theta: np.ndarray) -> Program:
    """
    @author: Joel
    Creates a program to set up an arbitrary state.
    :param theta: Vector representing the state.
    :return: PyQuil program setting up the state.
    """
    theta = np.concatenate((np.array([1]) / np.sqrt(theta.size + 1), theta),
                           axis=0)
    return create_arbitrary_state(theta)


def one_particle_ucc(dim, reference):
    """
    @author: Joel
    UCC-style ansatz preserving particle number.

    :param int dim: dimension of the space = num_qubits
    :param int reference: the binary number corresponding to the reference
        state (which must be a Fock-state).
    :return: function(theta) which returns the ansatz Program
    """
    # TODO: should this function also return the expected length of theta?
    terms = [[], []]
    for occupied in range(dim):
        if reference & (1 << occupied):
            for unoccupied in range(dim):
                if not reference & (1 << unoccupied):
                    term = FermionOperator(((unoccupied, 1), (occupied, 0))) \
                           - FermionOperator(((occupied, 1), (unoccupied, 0)))
                    term = qubitop_to_pyquilpauli(jordan_wigner(term))
                    assert len(term) == 2, "Term has not length two!"
                    terms[0].append(term[0])
                    terms[1].append(term[1])
    map_0 = exponential_map_commuting_pauli_terms(terms[0])
    map_1 = exponential_map_commuting_pauli_terms(terms[1])

    def wrap(theta):
        """
        Returns the ansatz Program.

        :param np.ndarray theta: parameters
        :return: the Program
        :rtype: pyquil.Program
        """
        return map_0(theta) + map_1(theta)

    return wrap


def multi_particle_ucc(dim):
    """
    @author: Joel
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

    :param int dim: dimension of the space = num_qubits**2
    :return: function(theta) which returns the ansatz Program. -1j*theta[i] is
        the coefficient in front of the term prod_k X_k^bit(i,k) where
        bit(i, k) is the k'th bit of i in binary, in the exponent.
    """
    # TODO: should this function also return the expected length of theta?
    terms = []
    for state in range(dim):
        term = QubitOperator(())
        for qubit in range(int.bit_length(state)):
            if state & (1 << qubit):
                term *= QubitOperator((qubit, 'X'))
        term = qubitop_to_pyquilpauli(term)
        assert len(term) == 1, "Term has not length one!"
        terms.append(term[0])
    return exponential_map_commuting_pauli_terms(terms)


def exponential_map_commuting_pauli_terms(terms):
    """
    @author = Joel
    Returns a function f(theta) which, given theta, returns the Program
    corresponding to exp(-1j sum_i theta[i]*term[i]) =
    prod_i exp(-1j*theta[i]*term[i]). Note that the equality only holds if
    the terms are commuting. This was inspired by pyquil.exponential_map
    and pyquil.exponentiate_commuting_pauli_sum.

    :param list of pyquil.paulis.PauliTerm terms: a list of pauli terms
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