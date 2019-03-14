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
from ansatz import one_particle_ansatz,one_particle_inital
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


def smallest_eig(H, ansatz, num_samples=None, opt_algorithm='L-BFGS-B'):
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
        eigval = H[0, 0]
        eigvect = np.array([1])
    return eigval, eigvect


def smallest_eig_vqe(H, qc_qvm, ansatz=None, num_samples=None, new_version=True,
                     opt_algorithm='Nelder-Mead', initial=None, maxiter=10000,
                     disp_run_info=False, display_after_run=False,
                     xatol=1e-2, fatol=1e-3, return_all_data=False):
                     
    """
    TODO: Fix this documentation. Below is not up to date.

    Finds the smallest eigenvalue and corresponding -vector of H using VQE.
    :param H: np.array hamiltonian matrix
    :param ansatz: ansatz function
    :param num_samples: number of samples on the qvm
    :param opt_algorithm:
    :param initial_params: ansatz parameters
    :return: list of energies
    """
    #if initial_params is None:
        #initial_params = 1/np.sqrt(H.shape[0])*np.array([1 for i in range(H.shape[0]-1)])


    if initial is None:
        initial = one_particle_inital(H.shape[0])

    if ansatz is None:
        ansatz = one_particle_ansatz

    # All options to Nelder-Mead
    disp_options = {'disp': display_after_run, 'xatol': xatol, 'fatol': fatol, 
                    'maxiter': maxiter}
    
    vqe = VQE(minimizer=minimize, minimizer_kwargs={'method': opt_algorithm, 
                                                    'options': disp_options})
    H = matrix_to_operator_1(H)

    # If disp_run_info is True we will print every step of the Nelder-Mead
    if disp_run_info: print_option = print
    else: print_option = lambda x:None

    if new_version:
        print(initial)
        eig = vqe.vqe_run(ansatz, H, initial, samples=num_samples, qc=qc_qvm,
                          disp=print_option, return_all=True)
    else:
        eig = vqe.vqe_run(ansatz, H, initial, samples=num_samples, qvm=qc_qvm, 
                          disp=print_option, return_all=True)

    #If option return_all_data is True we return a dict with data from all runs
    if return_all_data: 
        return eig
    else: 
        eigval = eig['fun']
        eigvect = eig['x']/np.linalg.norm(eig['x'])
        return eigval, eigvect


def calculate_negative_eigenvalues_vqe(H, ansatz, qvm, num_eigvals=None, num_samples=None, 
                                       opt_algorithm='L-BFGS-B', initial_params=None):
    """
    Calculates all negative or specified amount of eigenvalues for a given hamiltonian matrix.
    :param H: np.array hamiltonian matrix
    :param ansatz: ansatz function
    :param num_eigvals: number of desired eigenvalues to be calculated
    :param num_samples: number of samples on the qvm
    :param opt_algorithm:
    :param initial_params: ansatz parameters
    :return: list of energies
    @author: Eric Nilsson
    """
    if num_eigvals is None:
        num_eigvals = H.shape[0]
    energy = []
    for i in range(num_eigvals):
        eigval, eigvect = smallest_eig_vqe(H, ansatz, qvm, num_samples, opt_algorithm, initial_params)
        if eigval >= 0:
            if num_eigvals != H.shape[0]:
                print('Warning: Unable to find the specified amount of eigenvalues')
            return energy
        else:
            energy.append(eigval)
            # Maybe eigvect should be normalized??
            H = H + 1.1 * np.abs(energy[i]) * np.outer(eigvect, eigvect)  # move found eigenvalue to > 0.
    return energy


def calculate_eigenvalues_vqe(H, ansatz,qvm, num_eigvals=None, num_samples=None,
                              opt_algorithm='L-BFGS-B', initial_params=None):
    """
    Calculates all or specified amount of eigenvalues for an Hamiltonian matrix
    TODO: Make so it handles sparse matrices? Currently finds double zero eigenvalues
    :param H: np.array hamiltonian matrix
    :param ansatz: ansatz function
    :param num_eigvals: number of desired eigenvalues to be calculated
    :param num_samples: number of samples on the qvm
    :param opt_algorithm:
    :param initial_params: ansatz parameters
    :return: list of energies
    @author: Eric
    """
    energy = calculate_negative_eigenvalues_vqe(H, ansatz, qvm, num_eigvals, num_samples, opt_algorithm, initial_params)
    
    if num_eigvals is not None and len(energy) < num_eigvals:
        energy = energy + [-x for x in calculate_negative_eigenvalues_vqe(-1*H, ansatz, qvm, num_eigvals-len(energy),
                                                                          num_samples, opt_algorithm, initial_params)]
    if len(energy) < H.shape[0]:
        energy = energy + [-x for x in calculate_negative_eigenvalues_vqe(-1*H, ansatz, qvm, num_eigvals, num_samples,
                                                                          opt_algorithm, initial_params)]
        for i in range(len(energy),H.shape[0]): energy.append(0)


    return energy


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


