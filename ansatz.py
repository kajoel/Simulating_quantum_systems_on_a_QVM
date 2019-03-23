"""
Created on Mon Mar  4 10:50:49 2019
"""

# Imports
import numpy as np
from pyquil.quil import Program
from grove.alpha.arbitrary_state.arbitrary_state import create_arbitrary_state
from pyquil.paulis import exponential_map, PauliSum
from openfermion import QubitOperator
from forestopenfermion import qubitop_to_pyquilpauli


def one_particle(theta: np.ndarray) -> Program:
    """
    @author: Joel, Carl
    Creates a program to set up an arbitrary one-particle-state.
    :param theta: Vector representing the state.
    :return: PyQuil program setting up the state.
    """
    # TODO: doc behaves weird, theta: Union[ndarray, ndarray]

    vector = np.zeros(2 ** (theta.shape[0] + 1))
    vector[1] = 1
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
    return create_arbitrary_state(theta)


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
    terms = []
    for state in range(dim):
        term = QubitOperator(())
        for qubit in range(int.bit_length(state)):
            if state & (1 << qubit):
                term *= QubitOperator((qubit, 'X'))
        terms.append(qubitop_to_pyquilpauli(term)[0])
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


################################################################################
# TESTS
################################################################################
def _test_depth(ansatz, n_min=1, n_max=12, m=5):
    nn = n_max - n_min + 1
    nbr_ops = np.zeros(nn)
    for i in range(nn):
        n = n_min + i
        print(n)
        temp = np.empty(m)
        for j in range(m):
            temp[j] = len(ansatz(np.random.randn(n)))
        nbr_ops[i] = np.average(temp)
    return nbr_ops


################################################################################
# MAIN
################################################################################
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    nbr_ops_s = _test_depth(one_particle)
    nbr_ops_m = _test_depth(multi_particle, n_max=2 ** 5)

    plt.figure(0)
    plt.plot(nbr_ops_s)
    plt.title("Number of gates in one-particle-ansatz")

    plt.figure(1)
    plt.plot(nbr_ops_m)
    plt.title("Number of gates in multi-particle-ansatz")
