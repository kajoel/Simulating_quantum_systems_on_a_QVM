#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 08:14:54 2019

@author: kajoel
"""
import scipy.sparse as sparse
import numpy as np
from openfermion.ops import FermionOperator
from openfermion.ops import QubitOperator
from forestopenfermion import qubitop_to_pyquilpauli
from openfermion.transforms import jordan_wigner


###############################################################################
# VQE_RELATED FUNCTIONS
###############################################################################

"""
Creates a Hamiltonian operator from a (sparse) matrix. This function uses
a one-particle formulation and, thus, requires N qubits for an N-dimensional
Hilbert space.

@author: axelnathanson
"""
def matrix_to_operator_1(H):
    """Generates a PauliSum(pyquil) given a Hamiltonian-matrix
    
    Arguments:
        H [sparse.matrix] -- Hamiltonian matrix
    
    Returns:
        [PauliSum] -- Matrix representation of pauli-operators in pyquil
    """
    # Create the Hamiltonian with a and a^dagger
    Hamiltonian = FermionOperator()
    if sparse.issparse(H):
        H = H.tocoo()
        for i, j, data in zip(H.row, H.col, H.data):
            Hamiltonian += data * FermionOperator(((int(i), 1), (int(j), 0)))
    else:
        for i in range(H.shape[0]):
            for j in range(H.shape[1]):
                Hamiltonian += H[i,j]*FermionOperator(((i,1), (j,0)))

    Hamiltonian = jordan_wigner(Hamiltonian)
    Hamiltonian = qubitop_to_pyquilpauli(Hamiltonian)
    return Hamiltonian


def matrix_to_operator_2(H):
    """
    Creates a Qubit-operator from a (sparse) matrix. This function uses
    (almost) all states and, thus, requires (approximately) log(N) quibits
    for an N-dimensional Hilbert space.

    The idea for converting a matrix element to an operator is to raise/lower
    the qubits differing between two states to convert one basis-state to the
    other. The qubits that are not negated must be checked to have the correct
    value (using an analogue to the counting operator) to not get extra terms
    (one could perhaps allow for extra terms and compensate for them later?).

    0 = up
    1 = down

    (1+Zi)/2 checks that qubit i is 0 (up)
    (1-Zi)/2 checks that qubit i is 1 (down)
    (Xi+1j*Yi)/2 lowers qubit from 1 (down) to 0 (up)
    (Xi-1j*Yi)/2 raises qubit from 0 (up) to 1 (down)

    @author: kajoel
    """
    # Convert to sparse coo_matrix
    if not sparse.issparse(H):
        H = sparse.coo_matrix(H)
    elif H.getformat() != "coo":
        H = H.tocoo()
    # The main part of the function
    H_op = QubitOperator()
    for i, j, data in zip(H.row, H.col, H.data):
        new_term = QubitOperator(())  # = I
        for qubit in range(int.bit_length(H.shape[0] - 1)):
            if (i ^ j) & (1 << qubit):
                # lower/raise qubit
                new_term *= QubitOperator((qubit, "X"), 1 / 2) + \
                            QubitOperator((qubit, "Y"),
                                          1j * (int(j & (1 << qubit) != 0) - 1 / 2))
            else:
                # check that qubit has correct value (same as i and j)
                new_term *= QubitOperator((), 1 / 2) + \
                            QubitOperator((qubit, "Z"),
                                          1/2-int(j & (1 << qubit) != 0))
        H_op += data*new_term
    return qubitop_to_pyquilpauli(H_op)


###############################################################################
# TEST FUNCTIONS
###############################################################################

def _test_mat_to_op(hamiltonian_operator, jmin=0.5, jmax=100, tol=1e-8):
    """
    Tests that eigenvalues computed using a function that generates hamiltonian
    operators are correct. This test doesn't check for extra eigenvalues but
    only that the ones that should be there is there.
    @author: kajoel
    """
    import sys
    sys.path.insert(0, './')
    from lipkin_quasi_spin import hamiltonian, eigenvalues
    from openfermion.transforms import get_sparse_operator

    no_error = True
    for j2 in range(round(2 * jmin), round(2 * jmax) + 1):
        j = j2 / 2
        print("j = " + str(j))
        V = float(np.random.randn(1))
        H = hamiltonian(j, V)
        E = eigenvalues(j, V)
        for i in range(len(H)):
            H_op = hamiltonian_operator(H[i])
            H_op = get_sparse_operator(H_op).toarray()
            E_op = np.linalg.eigvals(H_op)
            # Check that E_op contains all eigenvalues in E[i]
            for E_ in E[i]:
                if all(abs(E_op - E_) > tol):
                    no_error = False
                    print("Max diff: " + str(max(abs(E_op - E_))))

    if no_error:
        print("Success!")
    else:
        print("Fail!")


###############################################################################
# MAIN
###############################################################################
if __name__ == "__main__":
    pass
    # The following tests has (successfully) been completed:
    # _test_mat_to_op(matrix_to_operator_1)
    # _test_mat_to_op(matrix_to_operator_1, jmax=11)

# get_sparse_operator seems to permute the basis. For 2 qubits:
# the permutation is (0 2 1 3) and for 3 qubits it's (0 4 2 6 1 5 3 7)
# or, in binary, (00, 10, 01, 11) and (000, 100, 010, 110, 001, 101, 011, 111).
