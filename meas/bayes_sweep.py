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
                 'func_evals': data_[3], 'facit': facit}

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
                          disp_data_during_run=False,
                          disp_progress=False,
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
    data_ = np.zeros( (5, len(num_evals)) )
    run_data = []

    # Calculates a facit with sample=None
    facit= vqe_eig.smallest(H, qc, init_params.alternate(dim_h),ansatz_, 
                            disp_run_info=False)
    

    for i,num_func_eval in enumerate(num_evals):
        if disp_progress:
            print('Number of function evaluations: {}'.format(num_func_eval))

        
        run_data.append(vqe_eig.smallest_bayes(H, qc, dimension, ansatz_, 
                                               samples, return_all_data=True, 
                                               n_calls= num_func_eval, 
                                               disp=disp_data_during_run))
        

        data_[0,i] = run_data[-1]['fun']
        data_[1,i] = np.linalg.norm(run_data[-1]['x']-facit[1])
        data_[2,i] = run_data[-1]['expectation_vars']
        data_[3,i] = num_func_eval
        data_[4,i] = num_func_eval * len(H) * samples
        
        if disp_progress:
            print('Done with calculation: {}/{}'.format(i+1,len(num_evals)))


    save_dict = {'samples': samples, 'exp_val': data_[0], 
                 'para_error': data_[1], 'variance': data_[2], 
                 'func_evals': data_[3], 'facit': facit, 
                 'quantum_evals': data_[4], 'all_data': run_data}

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
    if data.__contains__('quantum_evals'):
        data_x = data['quantum_evals']
    else:
        data_x = data['func_evals']


    start = data_x[0]
    stop = data_x[-1]
    data['variance'] = 2*np.sqrt(data['variance'])
    
    if label is None: label =  'Check metadata for ansatz'   
    i=0
    for key in data:
        if key == 'exp_val':
            plt.figure(i)
            plt.hlines(data['facit'][0], start, stop, colors='r', 
                       linestyles='dashed', 
                       label='True eig: {}'.format(round(data['facit'][0],4)))
    
            plt.errorbar(data_x,data[key],data['variance'], 
                         fmt='o', label=label, capsize=5)
            plt.legend()
            plt.xlabel('Function evaluations on the quntum computer')
            plt.ylabel(key)
            i+=1
        elif key == 'para_error':
            plt.figure(i)
            plt.hlines(0, start, stop, colors='r',linestyles='dashed')
            plt.scatter(data_x,data[key],label=label)
            plt.xlabel('Function evaluations on the quntum computer')
            plt.ylabel(key)
            plt.legend()
            i+=1

def heatmap_from_data():
    
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
                              samples=None, matrix_num=0, label = None, 
                              save = False, file_name = None, start=10, stop=40,
                              steps=5):
    
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
    
    return bayes_iteration_sweep(H, qc, ansatz_, h.shape[0], samples=samples,
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

def heatmap(ansatz_, convert_op, h, label = None, save = False,
            sample_start=100, sample_stop=1000, sample_step=10, 
            func_start=10, func_stop=40, func_steps=5, file_name=None):

    samples_sweep = range(sample_start, sample_stop, 
                    round((sample_stop-sample_start)/sample_step))
    func_eval = range(func_start, func_stop, 
                    round((func_stop-func_start)/func_steps))    

    dim = (len(samples_sweep),len(func_eval))
    data_ = np.zeros(dim)
    sample_mesh = np.zeros(dim)
    func_eval_mesh = np.zeros(dim)
    variance_mesh = np.zeros(dim)


    for index,sample in enumerate(samples_sweep):
        temp_data = run_bayes_iteration_sweep(ansatz_, convert_op, h=h,label=label, 
                                  steps=func_steps, start=func_start, 
                                  stop=func_stop, samples=sample)
        sample_mesh[index] += sample
        func_eval_mesh[index] = func_eval
        data_[index] = temp_data['para_error']
        variance_mesh[index] = temp_data['variance']
        print('Done with calculation: {}/{}'.format(index+1,len(samples_sweep)))



    if save:
        ansatz_name = ansatz_.__name__
        H = convert_op(h)
        if convert_op is matrix_to_op.one_particle: qubit = h.shape[0]
        elif convert_op is matrix_to_op.multi_particle:
            qubit = int.bit_length(h.shape[0])

        save_dict = {'para_error': data_, 'sample_mesh': sample_mesh,
                     'func_eval_mesh': func_eval_mesh, 
                     'variance': variance_mesh}
        
        metadata = {'ansatz': ansatz_name, 'Hamiltonian': H,
                    'minimizer': 'Bayesian Optimizer','Quibits': qubit,
                    'Type_of_meas': 'Sweep over samples with Bayesian optimizer', 
                    }

        data.save(file = file_name, data=save_dict,metadata=metadata)

    plt.figure()
    plt.imshow(data_, cmap='hot', interpolation='nearest')






################################################################################
# Main
################################################################################
if __name__ == '__main__':
    ansatz_ = ansatz.one_particle_ucc
    convert_op = matrix_to_op.one_particle
    #run_bayes_sample_sweep(ansatz_, convert_op, steps=5, stop=1000)
    #run_bayes_iteration_sweep(ansatz_, convert_op, steps=30, stop=100, save=True)
    #multiple_ansatzer()
    h = lipkin_quasi_spin.hamiltonian(1, 1)[0]
    
    heatmap(ansatz_, convert_op, h, save=True,
            sample_step=10, sample_start=100, sample_stop=1000, 
            func_steps=10, func_start=10, func_stop=100)
    
    plt.show()



