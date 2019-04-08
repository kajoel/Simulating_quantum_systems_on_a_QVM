#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fre 9 April 17:19 2019

@author: axel
The moduel contains methods that sweep over different parameters and in every
Optimizes the function with a Bayesian Optimization algorithm. 
The run can be saved, or plotted with the help functions. 


The test-cases can be used to run the Main part of the module, or to just plot 
data that has been generated with this module.

Disclaimer: This is a work in progress, the two functions will be turned into 
one, the current state works, but is far from idele. 
"""


# Imports for the module
from core import data, vqeOverride, vqe_eig, init_params
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import numpy as np

# Imports to test my methods
from core import ansatz, matrix_to_op, lipkin_quasi_spin
from pyquil import get_qc
from constants import ROOT_DIR
from os.path import join


def bayes_sample_sweep(H, 
                       qc, 
                       ansatz_, 
                       dim_h,
                       num_func_eval,
                       start = 1000, stop = 2000, steps = 3, 
                       save_after_run=False, 
                       plot_after_run=True,
                       label = None,
                       ansatz_name=None, 
                       qubits = None,
                       file_name = None):
    '''Runs a sweep over a number of samples, defined by start, 
       stop steps. For each given sample performs a Bayesian optimization. 
       Saves all data from the run and returns it as a dictionary. 
    
    Arguments:
        H -- Hamiltonian paulisum
        qc -- quantumcomputer
        ansatz_ -- given ansatz
        dim_h -- dimension of the hamiltonian.
        num_func_eval --  Number of funcion evaluations in the Bayesian Opt Alg.
    
    Keyword Arguments:
        start {int} -- Sample to start sweep on (default: {1000})
        stop {int} -- Sample to stop sweep on (default: {2000})
        steps {int} -- Number of steps in sweep (default: {3})
        save_after_run {bool} -- Saves run if True (default: {False})
        plot_after_run {bool} -- Plots the run if True (default: {True})
        label {str} -- Label for the plot (default: {None})
        
        Below is variables for the metadata if save_after_run is True.
        ansatz_name {str} -- Name of the ansatz (default: {None})
        qubits {int} -- Number of qubits (default: {None})
        file_name {str} -- Name of save file, if None: enter name into terminal 
    
    Returns:
        dict -- Dict with the data from all the runs.
    '''
    
    dimension = [(-1.0, 1.0) for i in range(dim_h-1)]
    

    # Sets up samples, and all vectors to save run in 
    samples = range(start, stop, round((stop-start)/steps))
    data_ = np.zeros( (4, len(samples)) )
    run_data = []

    # Calculates a facit with sample=None
    facit= vqe_eig.smallest(H, qc, init_params.alternate(dim_h),ansatz_)
    

    for i,sample in enumerate(samples):
        print('Number of samples: {}'.format(sample))
        
        run_data.append(vqe_eig.smallest_bayes(H, qc, dimension, ansatz_, 
                                               sample, return_all_data=True, 
                                               n_calls= num_func_eval))
        
        data_[0,i] = run_data[-1]['fun']
        data_[1,i] = np.linalg.norm(run_data[-1]['x']-facit[1])
        data_[2,i] = run_data[-1]['expectation_vars']
        data_[3,i] = num_func_eval
        print('Done with calculation: {}/{}'.format(i+1,len(samples)))


    save_dict = {'samples': samples, 'exp_val': data_[0], 
                 'para_error': data_[1], 'variance': data_[2], 
                 'func_evals': data_[3], 'facit': facit, 
                 'all_data': run_data}

    if save_after_run:
        metadata = {'ansatz': ansatz_name, 'Hamiltonian': H,
                    'minimizer': 'Bayesian Optimizer','Quibits': qubits,
                    'Type_of_meas': 'Sweep over samples with Bayesian optimizer', 
                    }

        data.save(file = file_name, data=save_dict,metadata=metadata)
    
    if plot_after_run: 
        if label is None: label = ansatz_name
        plot_sample_run(save_dict, label=label)
    

    return save_dict
    
def bayes_iteration_sweep(H, 
                          qc, 
                          ansatz_, 
                          dim_h,
                          samples = None,
                          start = 10, stop = 50, steps = 2, 
                          save_after_run=False, 
                          plot_after_run=True,
                          label = None,
                          ansatz_name=None, 
                          qubits = None,
                          file_name = None):
    '''Runs a sweep over a number of function evaluations, defined by start, 
       stop steps. For each given sample performs a Bayesian optimization. 
       Saves all data from the run and returns it as a dictionary. 
    
    Arguments:
        H -- Hamiltonian paulisum
        qc -- quantumcomputer
        ansatz_ -- given ansatz
        dim_h -- dimension of the hamiltonian.
    
    Keyword Arguments:
        samples -- Number of samples the qvm make per measurement.
        start {int} -- Sample to start sweep on (default: {1000})
        stop {int} -- Sample to stop sweep on (default: {2000})
        steps {int} -- Number of steps in sweep (default: {3})
        save_after_run {bool} -- Saves run if True (default: {False})
        plot_after_run {bool} -- Plots the run if True (default: {True})
        label {str} -- Label for the plot (default: {None})
        
        Below is variables for the metadata if save_after_run is True.
        ansatz_name {str} -- Name of the ansatz (default: {None})
        qubits {int} -- Number of qubits (default: {None})
        file_name {str} -- Name of save file, if None: enter name into terminal 
    
    Returns:
        dict -- Dict with the data from all the runs.
    '''
    
    dimension = [(-1.0, 1.0) for i in range(dim_h-1)]


    # Sets up samples, and all vectors to save run in 
    num_evals = range(start, stop, round((stop-start)/steps))
    data_ = np.zeros( (4, len(num_evals)) )
    run_data = []

    # Calculates a facit with sample=None
    facit= vqe_eig.smallest(H, qc, init_params.alternate(dim_h),ansatz_)
    

    for i,num_func_eval in enumerate(num_evals):
        print('Number of function evaluations: {}'.format(num_func_eval))
        
        run_data.append(vqe_eig.smallest_bayes(H, qc, dimension, ansatz_, 
                                               samples, return_all_data=True, 
                                               n_calls= num_func_eval))
        
        data_[0,i] = run_data[-1]['fun']
        data_[1,i] = np.linalg.norm(run_data[-1]['x']-facit[1])
        data_[2,i] = run_data[-1]['expectation_vars']
        data_[3,i] = num_func_eval
        print('Done with calculation: {}/{}'.format(i+1,len(num_evals)))


    save_dict = {'samples': samples, 'exp_val': data_[0], 
                 'para_error': data_[1], 'variance': data_[2], 
                 'func_evals': data_[3], 'facit': facit, 
                 'all_data': run_data}

    if save_after_run:
        metadata = {'ansatz': ansatz_name, 'Hamiltonian': H,
                    'minimizer': 'Bayesian Optimizer','Quibits': qubits,
                    'Type_of_meas': 'Sweep over func evals with Bayesian optimizer', 
                    }

        data.save(file = file_name, data=save_dict,metadata=metadata)

    if plot_after_run: 
        if label is None: label = ansatz_name
        plot_iteration_run(save_dict, label=label)
    

    return save_dict
    



################################################################################
# Extra methods
################################################################################

# Help method that plots a run
def plot_sample_run(data, label=None):
    """Plot data from a run of the method above.
    Arguments:
        data {dict} -- Dictionary generated with the method above. 
    
    Keyword Arguments:
        label {srt} -- Label for the legend (default: {None})
    """
    data['variance'] = 2*np.sqrt(data['variance'])
    start = data['samples'][0]
    stop = data['samples'][-1]
    
    if label is None: label =  'Check metadata for ansatz'   
    i=0
    for key in data:
        if key == 'facit' or key == 'samples' or key == 'all_data': continue
        elif key == 'exp_val':
            plt.figure(i)
            plt.hlines(data['facit'][0], start, stop, colors='r', 
                       linestyles='dashed', 
                       label='True eig: {}'.format(round(data['facit'][0],4)))
    
            plt.errorbar(data['samples'],data[key],data['variance'], 
                         fmt='o', label=label, capsize=5)
            plt.legend()
            plt.xlabel('Samples')
            plt.ylabel(key)
            i+=1
        else:
            plt.figure(i)
            if key == 'para_error':         
                plt.hlines(0, start, stop, colors='r',linestyles='dashed')
            plt.scatter(data['samples'],data[key],label=label)
            plt.xlabel('Samples')
            plt.ylabel(key)
            plt.legend()
            i+=1

def plot_iteration_run(data, label=None):
    """Plot data from a run of the iteration-sweep above.
    Arguments:
        data {dict} -- Dictionary generated with the method above. 
    
    Keyword Arguments:
        label {srt} -- Label for the legend (default: {None})
    """
    
    start = data['func_evals'][0]
    stop = data['func_evals'][-1]
    data['variance'] = 2*np.sqrt(data['variance'])
    
    if label is None: label =  'Check metadata for ansatz'   
    i=0
    for key in data:
        if key == 'exp_val':
            plt.figure(i)
            plt.hlines(data['facit'][0], start, stop, colors='r', 
                       linestyles='dashed', 
                       label='True eig: {}'.format(round(data['facit'][0],4)))
    
            plt.errorbar(data['func_evals'],data[key],data['variance'], 
                         fmt='o', label=label, capsize=5)
            plt.legend()
            plt.xlabel('Function evaluations')
            plt.ylabel(key)
            i+=1
        elif key == 'para_error':
            plt.figure(i)
            plt.hlines(0, start, stop, colors='r',linestyles='dashed')
            plt.scatter(data['func_evals'],data[key],label=label)
            plt.xlabel('Function evaluations')
            plt.ylabel(key)
            plt.legend()
            i+=1


################################################################################
# Tests of the methods / Measurments made with the module
################################################################################
def run_bayes_sample_sweep(ansatz_, convert_op, h=None, j=1, V=1, matrix_num=0, 
                        label = None, save = False, file_name = None,
                        start=100, stop=6000, steps=30):
    if h is None:
        h = lipkin_quasi_spin.hamiltonian(j, V)[matrix_num]

    ans_name = ansatz_.__name__
    ansatz_ = ansatz_(h.shape[0])
    if label is None: label = ans_name
    
    
    if convert_op is matrix_to_op.one_particle: qubit = h.shape[0]
    elif convert_op is matrix_to_op.multi_particle:
        qubit = qubits = int.bit_length(h.shape[0])
    else:
        print('Unknown ansatz')
        return

    qc = get_qc('{}q-qvm'.format(qubits))
    H = convert_op(h)
    
    return bayes_sample_sweep(H, qc, ansatz_, h.shape[0], num_func_eval=30,
                              start=start, stop=stop, steps=steps, 
                              save_after_run=save, label=label, qubits=qubit,
                              ansatz_name=ans_name, file_name=file_name)


def run_bayes_iteration_sweep(ansatz_, convert_op, h=None, j=1, V=1, 
                              matrix_num=0, label = None, save = False, 
                              file_name = None, start=10, stop=40, steps=5):
    
    if h is None:
        h = lipkin_quasi_spin.hamiltonian(j, V)[matrix_num]

    ans_name = ansatz_.__name__
    ansatz_ = ansatz_(h.shape[0])
    if label is None: label = ans_name
    
    
    if convert_op is matrix_to_op.one_particle: qubit = h.shape[0]
    elif convert_op is matrix_to_op.multi_particle:
        qubit = int.bit_length(h.shape[0])
    else:
        print('Unknown ansatz')
        return

    qc = get_qc('{}q-qvm'.format(qubit))
    H = convert_op(h)
    
    return bayes_iteration_sweep(H, qc, ansatz_, h.shape[0], samples=200,
                              start=start, stop=stop, steps=steps, 
                              save_after_run=save, label=label, qubits=qubit,
                              ansatz_name=ans_name, file_name=file_name)


def multiple_ansatzer(j=1, V=1, matrix_num=0):
    h_ = lipkin_quasi_spin.hamiltonian(j, V)[matrix_num]

    ansatz_ = ansatz.one_particle
    convert_op = matrix_to_op.one_particle
    run_bayes_iteration_sweep(ansatz_, convert_op, h=h_, label='One')

    ansatz_ = ansatz.one_particle_ucc
    run_bayes_iteration_sweep(ansatz_, convert_op, h=h_, label='One UCC')


    ansatz_ = ansatz.multi_particle
    convert_op = matrix_to_op.multi_particle
    run_bayes_iteration_sweep(ansatz_, convert_op, h=h_, label='Multi')


################################################################################
# Main
################################################################################
if __name__ == '__main__':
    ansatz_ = ansatz.one_particle_ucc
    convert_op = matrix_to_op.one_particle
    #run_bayes_sample_sweep(ansatz_, convert_op, steps=5, stop=1000)
    run_bayes_iteration_sweep(ansatz_, convert_op, steps=25, stop=100)
    #multiple_ansatzer()
    
    
    plt.show()



