#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 16:35:25 2019

@author: kajoel
"""

import numpy as np
import scipy.sparse as sparse

# Imports for VQE
import pyquil.api as api
from grove.pyvqe.vqe import VQE
from scipy.optimize import minimize
from ansatz import one_particle_ansatz
from matrix_to_operator import matrix_to_operator_1



###############################################################################
# MAIN VQE FUNCTIONS
###############################################################################
def calculate_eigenvalues(H, ansatz, update):
    """
    Calculates all eigenvalues of H using smallest_eig and update (to update
    Hamiltonian or ansatz to be able to find next eigenvalue).

    @author: kajoel
    """
    eigvals = np.empty(H.shape[0])
    for i in range(H.shape[0]):
        eigvals[i], eigvect = smallest_eig(H, ansatz)
        H, ansatz = update(H, ansatz, eigvals[i], eigvect)
    return eigvals


def smallest_eig(H, ansatz):
    """
    Finds the smallest eigenvalue and corresponding -vector of H using VQE.

    TODO

    Currently:
        assumes that H is a ndarray or sparse
        not using VQE (obviously)

    Should:
        be able to handle H as ndarray, sparse or operator (openfermion or pyquil)
        use VQE
    """
    if H.shape[0] > 1:
        eigval, eigvect = sparse.linalg.eigsh(H,1)
        eigvect = eigvect.reshape((eigvect.shape[0],))
        eigvect = eigvect/np.linalg.norm(eigvect)
    else:
        eigval = H[0,0]
        eigvect = np.array([1])
    return eigval, eigvect


'''
Finds the smallest eigenvalue and corresponding -vector of H using VQE.


'''
def smallest_eig_vqe(H, ansatz, num_samples=None, opt_algorithm = 'L-BFGS-B'):
    initial_value = np.array([1 for i in range(H.shape[0])])

    qvm = api.QVMConnection()
    vqe = VQE(minimizer=minimize, minimizer_kwargs={'method': opt_algorithm})
    H = matrix_to_operator_1(H)

    eig = vqe.vqe_run(ansatz, H, initial_value, samples=num_samples, qvm=qvm)
    eigval = eig['fun']
    eigvect = eig['x']/np.linalg.norm(eig['x'])

    return eigval,eigvect


###############################################################################
# UPDATE (HAMILTONIAN AND/OR ANSATZ) FUNCTIONS
###############################################################################
def update_householder(H,ansatz,_,x):
    """
    Updates the Hamiltonian by block diagonalization using a Householder transform
    to reduce the dimensionality by one in each step. This function were made for
    numpy-arrays (ndarrays) rather than sparse (which is preferred in
    matrix_to_operator_2). Npte: x should be normalized

    If we are going to use Householder transformations to reduce dimensionality
    I believe that ndarrays is the way to go, since householder-matrices are
    dense. Since this strategy requires matrix-multiplication it might be faster
    to add a x^H x to the Hamiltonian; at least if a x^H x can be turned into
    operators efficiently (currently I don't know anything better than mat2op_2).

    Note that if ||n|| is small there might be stabilty-issues.

    @author: kajoel
    """
    if x.shape[0]>1:
        # Find suitable basis-vector to reflect to (with householder)
        y = np.zeros(x.shape[0])
        idx = int(abs(x[0])>0.5)  # this makes sure that householder is stable
        y[idx] = 1

        # Create Householder-matrix T
        n = x-y
        n = n/np.linalg.norm(n)
        T = np.eye(n.shape[0]) - 2*np.outer(n,n)

        # Calculate new hamiltonian
        T.dot(H.dot(T, out=H), out=H)
        H = np.delete(H, idx, axis=0)
        H = np.delete(H, idx, axis=1)
    return H, ansatz


###############################################################################
# TEST FUNCTIONS
###############################################################################
"""
Tests that calculate_eigenvalues(H, None, update_householder) works.

NOTE: This test assumes an old and incomplete version of smallest_eig.
"""
def _test_1():
    import sys
    sys.path.insert(0, './')
    from lipkin_quasi_spin import hamiltonian, eigenvalues

    j = 4.5
    V = 1
    tol=1e-8
    no_error = True
    H = hamiltonian(j, V)
    E = eigenvalues(j, V)
    for i in range(len(H)):
        eigs = calculate_eigenvalues(H[i].toarray(), None, update_householder)
        for E_ in E[i]:
            if all(abs(eigs - E_) > tol):
                no_error = False
                print("Max diff: " + str(max(abs(eigs-E_))))
    if no_error:
        print("Success!")
    else:
        print("Fail!")

###############################################################################
# MAIN
###############################################################################
if __name__ == "__main__":
    _test_1()


