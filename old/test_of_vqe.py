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

# Change the days date if you want to save to CSV-file
from core import init_params

year = 19
month = 3
day = 13
datetime(year, month, day)

# Imports from our projects
from core.matrix_to_op import multi_particle
from core.lipkin_quasi_spin import hamiltonian
from core.ansatz import multi_particle as ansatz
from core.vqe_eig import smallest as vqe_eig, smallest


# Egentligen helt onödig nu efter jag skrivit om smallest, är i princip
# bara den funktionen med en print
def count_opt_iterations(H, qc_qvm, new_version=True, samples=None,
                         disp_run_info=False, return_dict=False, xatol=1e-2,
                         fatol=1e-3, maxiter=10000):
    """Count the number of iterations the Nelder-Mead takes to converge
    Arguments:
        :param H: Hamiltonian matrix
        :param qvm_qc: Quantum computer
    
    Keyword Arguments:
        :param old_version: bool if you are using a old version. default:False
        :param samples: int representing number of samples. default:None
        :param disp_run_info: bool, if true prints all intermidiate 
                              calculated values. default: False
        :param xatol: float, error in x_opt between Nelder-M iterations
        :param fatol: float, error in f(x_opt) between Nelder-M iterations
        :param maxiter: int, iterations before termination. default:10000
        :param return_dict: bool, if True return data of all iterations
    Returns:
        Numer of iterations, or the dict with data of all iterations.
        
    """
    result = vqe_eig(H, qc_qvm, ansatz, num_samples=samples,
                     new_version=new_version, display_after_run=True,
                     disp_run_info=disp_run_info, xatol=xatol, fatol=fatol,
                     maxiter=maxiter, return_all_data=True)

    print('Real eigs:')
    print(np.linalg.eigvals(H.toarray()))
    print('VQE Calculated eigenvalue:')
    print(result['fun'])

    if return_dict:
        return result
    else:
        return len(result['iteration_params'])


def sweep_parameters(H, qvm_qc, new_version=True, num_para=20, start=-10,
                     stop=10, samples=None, fig_nr=0, save=False, plot=True,
                     callback=None):
    '''
    TODO: Add a statement that saves the data from the run and comments.
    '''
    vqe = VQE(minimizer=minimize, minimizer_kwargs={'method': 'Nelder-Mead'})
    if H.shape[0] > 3:
        print('To many parameters to represent in 2 or 3 dimensions')
        return
    elif H.shape[0] is 1:
        print('Nothing to sweep over')
        return
    elif H.shape[0] is 2:
        H = multi_particle(H)
        parameters = np.linspace(start, stop, num_para)

        if new_version:
            exp_val = []

            for para in parameters:
                tmp = vqe.expectation(ansatz(np.array([para])), H,
                                      samples=samples, qc=qvm_qc)
                exp_val.append(tmp)
                if callback is not None: callback(para, tmp)
        else:
            exp_val = [vqe.expectation(ansatz(np.array([para])), H,
                                       samples=samples, qvm=qvm_qc)
                       for para in parameters]

        if (plot):
            plt.figure(fig_nr)
            plt.plot(parameters, exp_val, label='Samples: {}'.format(samples))
            plt.xlabel('Parameter value')
            plt.ylabel('Expected value of Hamiltonian')
            plt.show()
        return [parameters, exp_val]
    else:
        H = one_particle(H)
        exp_val = np.zeros((num_para, num_para))
        mesh_1 = np.zeros((num_para, num_para))
        mesh_2 = np.zeros((num_para, num_para))
        parameters = np.linspace(start, stop, num_para)

        for i, p_1 in enumerate(parameters):
            mesh_1[i] += p_1
            mesh_2[i] = parameters
            if new_version:
                exp_val[i] = [vqe.expectation(ansatz(np.array([p_1, p_2])), H,
                                              samples=samples, qc=qvm_qc)
                              for p_2 in parameters]
            else:
                exp_val[i] = [vqe.expectation(ansatz(np.array([p_1, p_2])), H,
                                              samples=samples, qvm=qvm_qc)
                              for p_2 in parameters]

        fig = plt.figure(fig_nr)
        ax = fig.add_subplot(111, projection='3d')
        # Plot the surface

        save_run_to_csv(exp_val)
        ax.plot_surface(mesh_1, mesh_2, exp_val, cmap=cm.coolwarm)
        return


def save_run_to_csv(Variable):
    """ Save given variable to CSV-file with name of exakt time
    Arguments:
        Variable{np.array}--Variable to save to .txt file in CSV-format
    """
    np.savetxt('{}'.format(datetime.now()), Variable)


################################################################################
# TESTS
################################################################################

def main1(samples=1000):
    qc = get_qc('3q-qvm')
    j, V = 1, 1
    H, _ = hamiltonian(j, V)
    result = count_opt_iterations(H, qc, new_version=True, samples=samples,
                                  fatol=1e-2, xatol=1e-3, return_dict=True,
                                  disp_run_info=True)

    plt.figure(1)
    plt.plot(result['iteration_params'], result['expectation_vals'])
    plt.show()


def main2(qc, j, H, samples=1000, sweep_params=100, callback=None, plot=False):
    sweep_parameters(H, qc, new_version=True, samples=samples,
                     num_para=sweep_params, start=-3, stop=3, callback=
                     callback, plot=plot)


################################################################################
# Main
################################################################################

if __name__ == '__main__':
    samples = 5000
    sweep_params = 30

    qc = get_qc('3q-qvm')
    j, V = 1, 1
    h = hamiltonian(j, V)[0]

    oldx = None
    oldy = None


    def testprint2(x, y):
        global oldx
        global oldy
        if oldx is not None and oldy is not None: plt.plot([oldx, x], [oldy, y],
                                                           color='red')
        oldx = x
        oldy = y
        plt.pause(0.05)


    main2(qc, j, h, samples, sweep_params,
          callback=testprint2)

    import matplotlib.pyplot as plt


    def testprint(x, y):
        plt.scatter(x, y, color='blue')
        plt.pause(0.05)
        print("Parameter: {}".format(x[0]))
        print("Expectation: {}".format(y))


    energies = smallest(multi_particle(h), qc, init_params.ones(h.shape[0]),
                        ansatz_=ansatz, samples=samples,
                        disp_run_info=testprint, fatol=1e-1*16/np.sqrt(samples))[0]
