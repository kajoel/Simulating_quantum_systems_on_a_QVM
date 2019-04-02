#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tus Mar 27 14:26 2019

@author: axel
"""
# Imports
from core import ansatz, matrix_to_op, init_params, lipkin_quasi_spin, vqe_eig, data, vqeOverride
from scipy.optimize import minimize
import numpy as np
from pyquil import get_qc
import matplotlib.pyplot as plt


# Main method
def error_of_sample(H, qc, ansatz_, dim_h, initial_p = init_params.alternate,
                    start=1000, stop=2000, steps=3,save_run=False,label_ = None, 
                    ansatz_name=None, fig_nr=0):

    vqe = vqeOverride.VQE_override(minimizer=minimize,
                               minimizer_kwargs={'method': 'Nelder-Mead'})

    # Sets up samples, and all vectors to save run in 
    samples = range(start, stop, round((stop-start)/steps))
    exp_val = np.zeros(len(samples))
    para_error = np.zeros(len(samples))
    variance = np.zeros(len(samples))
    iterations = np.zeros(len(samples))
    
    facit= vqe_eig.smallest(H, qc, initial_p(dim_h),ansatz_)
    
    # Main part of method, 
    for i,sample in enumerate(samples):
        print('Number of samples: {}'.format(sample))
        xatol_ = 1e-3

        # Makes one calculation with given samples, sets fatol accordingly.
        _, temp_var = vqe.expectation(ansatz_(initial_p(dim_h)), H,
                                        samples=sample,qc=qc)
        fatol_ = np.sqrt(temp_var)

        run_data = vqe_eig.smallest(H, qc, initial_p(dim_h), ansatz_,
                                    samples=sample, fatol=fatol_, xatol=xatol_,
                                    return_all_data=True)
        

        exp_val[i], temp_var = vqe.expectation(ansatz_(run_data['x']), H,
                                        samples=sample,qc=qc)

        iterations[i] = len(run_data['expectation_vals'])
        variance[i] = 2*np.sqrt(temp_var)
        para_error[i] = np.linalg.norm(run_data['x']-facit[1])
        print('Done with calculation: {}/{}'.format(i+1,len(samples)))

    if save_run: save_error_run(samples, exp_val, para_error, ansatz_name, 
                                H,variance, iterations)

    plot_error(facit, samples, exp_val, para_error, variance, iterations, start, stop, 
               label_, fig_nr)
    



################################################################################
# Extra methods
################################################################################

# Help-method that saves the run
def save_error_run(samples,exp_value,para_error,ansatz_, H,variance=None,
                   iterations=None):

    data_ = {'Samples' : samples,'Expected_values': exp_value,
             'Parameter_error': para_error,'Variance': variance, 
             'Iterations': iterations}

    metadata = {'ansatz': ansatz_, 'Hamiltonian': H,'minimizer': 'Nelder-Mead', 
                'Type_of_measurement': 'Eigenvalues, parametererrors and iterations for different samples.'}

    data.save(data=data_,metadata=metadata)


# Help method that plots a run
def plot_error(facit, samples, exp_val, para_error, variance, iterations,
               start, stop, label=None, fig_nr = 0):
    if label is None:
        label_1 = 'Eig calculated with Nelder-Mead'
        label_2 = 'Error of parameter'
        label_3 = 'Number of iterations'
    else:
        label_1, label_2, label_3 = label, label, label
    
    # First figure is with energies
    plt.figure(fig_nr)
    plt.hlines(facit[0], start, stop, colors='r', linestyles='dashed',
                   label='True eig: {}'.format(round(facit[0],4)))
    plt.errorbar(samples,exp_val,variance, fmt='o',
                 label=label_1)
    plt.legend()
    plt.xlabel('Samples')
    plt.ylabel('Eigenvalue')

    # Second with parameters
    plt.figure(fig_nr+1)
    plt.hlines(0, start, stop, colors='r', linestyles='dashed',
                   label='True parameter')
    plt.scatter(samples,para_error,label=label_2)
    plt.xlabel('Samples')
    plt.ylabel('Error in parameter')
    plt.legend()

    # Third one with iterations
    plt.figure(fig_nr+2)
    plt.scatter(samples,iterations,label=label_3)
    plt.xlabel('Samples')
    plt.ylabel('Number of iterations')
    plt.legend()



################################################################################
# Test
################################################################################
def run_sample_error(ansatz_, convert_op, j=1, V=1, matrix_num=0, 
                    label_ = None, fig_nr=0):
    
    h = lipkin_quasi_spin.hamiltonian(j, V)[matrix_num]

    ansatz_name = ansatz_.__name__
    ansatz_ = ansatz_(h.shape[0])
    if label_ is None: label_ = 'Dim of matrix: {}'.format(h.shape[0])
    h = lipkin_quasi_spin.hamiltonian(j, V)[matrix_num]
    
    if convert_op is matrix_to_op.one_particle: qubits = h.shape[0]
    elif convert_op is matrix_to_op.multi_particle:
        qubits = qubits = int.bit_length(h.shape[0])
    else:
        print('Unknown ansatz')
        return

    qc = get_qc('{}q-qvm'.format(qubits))
    H = convert_op(h)
    
    error_of_sample(H, qc, ansatz_, h.shape[0], start=100, stop=6000, steps=60, 
                    label_=label_, save_run=True, ansatz_name=ansatz_name, 
                    fig_nr=fig_nr)

def test_var(ansatz_, convert_op, j=1, V=1, matrix_num=0,):
    h = lipkin_quasi_spin.hamiltonian(j, V)[matrix_num]

    if convert_op is matrix_to_op.one_particle:
        qubits = h.shape[0]
    elif convert_op is matrix_to_op.multi_particle:
        qubits = qubits = int.bit_length(h.shape[0])
    else:
        print('Unknown ansatz')
        return

    ansatz_ = ansatz_(h.shape[0])
    initial_p = init_params.alternate
    h = lipkin_quasi_spin.hamiltonian(j, V)[matrix_num]
    qc = get_qc('{}q-qvm'.format(qubits))
    H = convert_op(h)
    
    sample=2000
    
    var = vqe_eig.smallest(H, qc, initial_p(h.shape[0]), ansatz_, 
                           samples=sample, fatol=1e-1,return_all_data=True)

    print(var['fun'])
    for key in var: 
        print(key)
        print(var[key])

################################################################################
# Main
################################################################################
if __name__ == '__main__':
    convert_op = matrix_to_op.multi_particle
    ansatz_ = ansatz.multi_particle


    j_=2
    run_sample_error(ansatz_,convert_op ,j=j_,matrix_num=1 ,fig_nr=j_*2)

    plt.show()



