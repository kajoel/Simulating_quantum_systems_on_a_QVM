#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mar  10 10:39 2019

@author: axelnathanson
"""
# Imports
import numpy as np
import pyquil.api as api
from scipy.optimize import minimize
from grove.pyvqe.vqe import VQE
from matplotlib import pyplot as plt
from datetime import datetime, date
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from pyquil import get_qc
import time

# Change the days date if you want to save to CSV-file
year = 19
month = 3
day = 13
datetime(year, month, day)

# Imports from our projects
from matrix_to_operator import matrix_to_operator_1
from lipkin_quasi_spin import hamiltonian, eigenvalues
from ansatz import one_particle_ansatz as ansatz
from ansatz import one_particle_inital as initial


def count_opt_iterations(H, qvm, samples=None, disp_opt=False,
                         compare_eig=False,
                         xatol=1e-2, fatol=1e-3, maxiter=10000):
    """Count the number of iterations the Nelder-Mead takes to converge
    
    Arguments:
        H {Matrix} -- Hamiltonian matrix
        qvm {quantum computer} -- QMVConnection as of right now
    
    Keyword Arguments:
        samples {int} -- Number of samples (default: {None})
        disp_opt {bool} --
        Display output in each step of optimization (default: {False})
        compare_eig {bool} --
        Compare vqe calculated eig with analytical (default: {False})
        xatol {float} --
        error in x_opt between iterations for convergence (default: {1e-2})
        fatol {float} --
        error in f(x_opt) between iterations for convergence (default: {1e-3})
        maxiter {int} --
        Maximum number of iterations before termination (default: {10000})
    Returns:
        iterations {int} -- Number of iterations before convergence
    """

    Option = {'disp': disp_opt, 'xatol': xatol, 'fatol': fatol,
              'maxiter': maxiter}
    vqe = VQE(minimizer=minimize,
              minimizer_kwargs={'method': 'Nelder-Mead', 'options': Option})
    H_program = matrix_to_operator_1(H)

    if compare_eig is True:
        # result = vqe.vqe_run(ansatz, H_program,
        # initial(H.shape[0]), samples=samples,
        # qc=qvm, disp=lambda x: None, return_all=True)
        result = vqe.vqe_run(ansatz, H_program, initial(H.shape[0]),
                             samples=samples,
                             qc=qvm, disp=print, return_all=True)

        print('Real eigenvalues:')
        print(np.linalg.eigvals(H.toarray()))
        print('VQE Calculated eigenvalue:')
        print(result['fun'])
        # return len(result['iteration_params'])
        return result
    else:
        # return len(vqe.vqe_run(
        # ansatz, H_program, initial(H.shape[0]), samples=samples,
        # qc=qvm, disp=lambda x: None, return_all=True)['iteration_params'])
        return len(
            vqe.vqe_run(ansatz, H_program, initial(H.shape[0]), samples=samples,
                        qc=qvm, disp=print, return_all=True)[
                'iteration_params'])


def swep_parameters(H, qvm, num_para=20, start=-10, stop=10, samples=None,
                    fig_nr=0, save=False):
    '''
    TODO: Add a statement that saves the data from the run, when the
     para save=True
    '''
    vqe = VQE(minimizer=minimize, minimizer_kwargs={'method': 'Nelder-Mead'})
    if H.shape[0] > 3:
        print('To many parameters to represent in 2 or 3 dimensions')
        return
    elif H.shape[0] is 1:
        print('Nothing to sweep over')
        return
    elif H.shape[0] is 2:
        H = matrix_to_operator_1(H)
        parameters = np.linspace(start, stop, num_para)
        exp_val = [vqe.expectation(ansatz(np.array([para])), H, samples=samples,
                                   qc=qvm) for para in parameters]

        plt.figure(fig_nr)
        plt.plot(parameters, exp_val)
        plt.xlabel('Paramter value')
        plt.ylabel('Expected value of Hamiltonian')
        # plt.show()
        return
    else:
        H = matrix_to_operator_1(H)
        exp_val = np.zeros((num_para, num_para))
        mesh_1 = np.zeros((num_para, num_para))
        mesh_2 = np.zeros((num_para, num_para))
        parameters = np.linspace(start, stop, num_para)

        for i, p_1 in enumerate(parameters):
            mesh_1[i] += p_1
            mesh_2[i] = parameters
            exp_val[i] = [vqe.expectation(ansatz(np.array([p_1, p_2])), H,
                                          samples=samples, qc=qvm) for p_2 in
                          parameters]

        fig = plt.figure(fig_nr)
        ax = fig.add_subplot(111, projection='3d')
        # Plot the surface

        ax.plot_surface(mesh_1, mesh_2, exp_val, cmap=cm.coolwarm)
        # plt.show()
        return


def save_run_to_csv(Variable):
    """ Save given variable to CSV-file with name of exakt time down to mili-sec
    Arguments:
        Variable {np.array} -- Variable you want to save to .txt file in
        CSV-format
    """

    np.savetxt('{}'.format(datetime.now()), Variable)


if __name__ == '__main__':
    # qvm = api.QVMConnection()
    qc = get_qc("2q-qvm")
    j, V = 1, 1
    H, _ = hamiltonian(j, V)
    print('Size of matrix:', H.shape[0])
    start = time.time()
    # swep_parameters(H, qc, samples=None)
    swep_parameters(H, qc, num_para=50, samples=10000)
    do = True
    if do:
        ans = count_opt_iterations(H, qc, compare_eig=True, samples=10000,
                                   disp_opt=True, xatol=1e-3, fatol=1e-2,
                                   maxiter=25)
        iteration_params = ans['iteration_params']
        funvals = ans['expectation_vals']
        plt.plot(iteration_params, funvals, 'ro-')
    end = time.time()
    print('Time taken:', end - start)
    plt.show()
