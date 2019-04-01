#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tus Mar 27 14:26 2019

@author: axel
"""
# Imports
from core import ansatz,matrix_to_op,init_params, lipkin_quasi_spin,vqe_eig,data
import numpy as np
from pyquil import get_qc
import matplotlib.pyplot as plt


def error_of_sample(H, qc, ansatz_, dim_h, initial_p = init_params.alternate,
                    start=1000, stop=2000, steps=3,save_run=False):

    samples = range(start, stop, round((stop-start)/steps))
    exp_val = np.zeros(len(samples))
    para_error = np.zeros(len(samples))
    variance = np.zeros(len(samples))


    facit= vqe_eig.smallest(H, qc, initial_p(dim_h),ansatz_)
    
    for i,sample in enumerate(samples):
        run_data = vqe_eig.smallest(H, qc, initial_p(dim_h), ansatz_,
                                    samples=sample, fatol=1e-2,
                                    return_all_data=True)
        
        variance[i] = np.sqrt(run_data['variance'][-1])
        exp_val[i] = run_data['fun']
        para_error[i] = np.linalg.norm(run_data['x']-facit[1])
        print('Done with calculation: {}/{}'.format(i+1,len(samples)))

    if save_run: save_error_run(samples, exp_val, para_error, ansatz_, 
                                H,variance)

    # Figure 0 is with energies
    plt.figure(0)
    plt.hlines(facit[0], start, stop, colors='r', linestyles='dashed',
                   label='True eig: {}'.format(round(facit[0],4)))
    plt.errorbar(samples,exp_val,variance, fmt='o',
                 label='Eig calculated with Nelder-Mead')
    plt.legend()
    plt.xlabel('Samples')
    plt.ylabel('Eigenvalue')

    # Figure 1 is with parameters
    plt.figure(1)
    plt.hlines(0, start, stop, colors='r', linestyles='dashed',
                   label='True parameter')
    plt.scatter(samples,para_error,label='Error of parameter')
    plt.xlabel('Samples')
    plt.ylabel('Error in parameter')
    plt.legend()


def save_error_run(samples,exp_value,para_error,ansatz_, H,variance=None):
    data_ = {'Samples' : samples,'Expected_values': exp_value,
             'Parameter_error': para_error,'Variance': variance}

    metadata = {'ansatz': ansatz_, 'Hamiltonian': H,'minimizer': 'Nelder-Mead', 
                'Type_of_measurement': 'Eigenvalues and parametererrors for different samples.'}

    data.save('ParaEigErrorsSweep',data_,metadata)




################################################################################
# Test
################################################################################
def run_sample_error(ansatz_, j=1, V=1, matrix_num=0,):
    h = lipkin_quasi_spin.hamiltonian(j, V)[matrix_num]
    if ansatz_ is ansatz.one_particle:
        convert_op = matrix_to_op.one_particle
        qubits = h.shape[0]
    elif ansatz_ is ansatz.multi_particle:
        convert_op = matrix_to_op.multi_particle
        qubits = int.bit_length(h.shape[0])
    else:
        print('Unknown ansatz')
        return
    
    ansatz_ = ansatz_(h.shape[0])
    h = lipkin_quasi_spin.hamiltonian(j, V)[matrix_num]
    if ansatz_ is ansatz.one_particle:
        convert_op = matrix_to_op.one_particle
        qubits = h.shape[0]
    elif ansatz_ is ansatz.multi_particle:
        convert_op = matrix_to_op.multi_particle
        qubits = int.bit_length(h.shape[0])
    else:
        print('Unknown ansatz')
        return
    
    qc = get_qc('{}q-qvm'.format(qubits))
    H = convert_op(h)
    error_of_sample(H, qc, ansatz_, h.shape[0], start=1000, stop=5000, steps=15)

def test_var(ansatz_, j=1, V=1, matrix_num=0,):
    h = lipkin_quasi_spin.hamiltonian(j, V)[matrix_num]
    if ansatz_ is ansatz.one_particle:
        convert_op = matrix_to_op.one_particle
        qubits = h.shape[0]
    elif ansatz_ is ansatz.multi_particle:
        convert_op = matrix_to_op.multi_particle
        qubits = int.bit_length(h.shape[0])
    else:
        print('Unknown ansatz')
        return
    
    initial_p = init_params.alternate
    h = lipkin_quasi_spin.hamiltonian(j, V)[matrix_num]
    if ansatz_ is ansatz.one_particle:
        convert_op = matrix_to_op.one_particle
        qubits = h.shape[0]
    elif ansatz_ is ansatz.multi_particle:
        convert_op = matrix_to_op.multi_particle
        qubits = int.bit_length(h.shape[0])
    else:
        print('Unknown ansatz')
        return
    
    qc = get_qc('{}q-qvm'.format(qubits))
    H = convert_op(h)
    
    sample=2000

    var = vqe_eig.smallest(H, qc, initial_p(h.shape[0]), ansatz_, 
                           samples=sample, fatol=1e-1,return_all_data=True)

    print(var['fun'])
    print(var['variance'])
    for key in var: 
        print(key)
        print(var[key])

################################################################################
# Main
################################################################################
if __name__ == '__main__':

    ansatz_ = ansatz.one_particle
    run_sample_error(ansatz_)
    plt.show()



