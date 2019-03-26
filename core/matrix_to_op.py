#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 08:14:54 2019

@author: kajoel
"""
import scipy.sparse as sparse
import numpy as np
import pyquil.paulis
from openfermion.ops import FermionOperator
from openfermion.ops import QubitOperator
from forestopenfermion import qubitop_to_pyquilpauli
from openfermion.transforms import jordan_wigner


###############################################################################
# VQE_RELATED FUNCTIONS
###############################################################################


def one_particle(H: sparse.coo_matrix) -> pyquil.paulis.PauliSum:
    """
    Generates a PauliSum(pyquil) given a Hamiltonian-matrix.

    Creates a Hamiltonian operator from a (sparse) matrix. This function uses
    a one-particle formulation and, thus, requires N qubits for an N-dimensional
    Hilbert space.

    @author: Axel, Joel

    :param H: An array (array_like) representing the hamiltonian.
    :return: Hamiltonian as PyQuil PauliSum.
    """
    # Create the Hamiltonian with a and a^dagger
    Hamiltonian = FermionOperator()
    if sparse.issparse(H):
        H = H.tocoo()
        for i, j, data in zip(H.row, H.col, H.data):
            Hamiltonian += data * FermionOperator(((int(i), 1), (int(j), 0)))
    else:
        if not isinstance(H, np.ndarray):
            H = np.asanyarray(H)
        for i in range(H.shape[0]):
            for j in range(H.shape[1]):
                Hamiltonian += H[i, j] * FermionOperator(((i, 1), (j, 0)))

    Hamiltonian = jordan_wigner(Hamiltonian)
    Hamiltonian = qubitop_to_pyquilpauli(Hamiltonian)
    return Hamiltonian


def multi_particle(H: sparse.coo_matrix) -> pyquil.paulis.PauliSum:
    """
    Creates a Qubit-operator from a (sparse) matrix. This function uses
    (almost) all states and, thus, requires (approximately) log(N) qubits
    for an N-dimensional Hilbert space.

    The idea for converting a matrix element to an operator is to raise/lower
    the qubits differing between two states to convert one basis-state to the
    other. The qubits that are not negated must be checked to have the correct
    value (using an analogue of the counting operator) to not get extra terms
    (one could perhaps allow for extra terms and compensate for them later?).

    0 = up
    1 = down

    (1+Zi)/2 checks that qubit i is 0 (up)
    (1-Zi)/2 checks that qubit i is 1 (down)
    (Xi+1j*Yi)/2 lowers qubit from 1 (down) to 0 (up)
    (Xi-1j*Yi)/2 raises qubit from 0 (up) to 1 (down)

    @author: Joel

    :param H: An array (array-like) representing the hamiltonian.
    :return: Hamiltonian as PyQuil PauliSum.
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
                                          1j * (int(
                                              j & (1 << qubit) != 0) - 1 / 2))
            else:
                # check that qubit has correct value (same as i and j)
                new_term *= QubitOperator((), 1 / 2) + \
                            QubitOperator((qubit, "Z"),
                                          1 / 2 - int(j & (1 << qubit) != 0))
        H_op += data * new_term
    return qubitop_to_pyquilpauli(H_op)

