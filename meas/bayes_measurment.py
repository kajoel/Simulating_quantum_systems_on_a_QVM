#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Ons 17 April 10:48 2019

@author: axel 
"""


# Imports for the module
from core import data, vqe_override, vqe_eig, init_params, create_vqe
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import numpy as np



# Imports to test my methods
from core import ansatz, matrix_to_op, lipkin_quasi_spin
from pyquil import get_qc
from constants import ROOT_DIR
from os.path import join



def bayes_iteration_sweep(h, 
                          qc, 
                          ansatz_, 
                          convert_op,
                          intervall = 5,
                          samples = None,
                          start = 15, stop = 45, steps = 10, 
                          measurments_per_step=3,
                          save_after_run=False, 
                          plot_after_run=True,
                          disp_data_during_run=True,
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
        measurments_per_step {int} -- Number of measurments per step in sweep
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


    dimension = [(float(-intervall), float(intervall)) for i in range(h.shape[0]-1)]
    H = convert_op(h)

    # Sets up samples, and all vectors to save run in 
    num_evals = range(start, stop, round((stop-start)/steps))
    data_ = np.zeros( (5, len(num_evals)) )
    run_data = []
    # Calculates a facit with sample=None
    
    vqe = create_vqe.default_nelder_mead()
    facit = vqe_eig.smallest(H, qc, init_params.alternate(h.shape[0]), vqe, 
                             ansatz_)
    
    all_data=[]
    for i,num_func_eval in enumerate(num_evals):
        vqe = create_vqe.default_bayes(n_calls=num_func_eval)
        
        temp_data=np.zeros( (3, measurments_per_step) )
        for j in range(measurments_per_step):
    
            run_data = vqe_eig.smallest(H, qc, dimension, vqe, ansatz_, 
                                        samples,  
                                        disp_run_info=disp_data_during_run)
    
            temp_data[0,j] = run_data['fun']
            temp_data[1,j] = np.linalg.norm(run_data['x']-facit[1])
            temp_data[2,j] = run_data['expectation_vars']

        all_data.append(temp_data)
        data_[0,i] = np.mean(temp_data[0])
        data_[1,i] = np.mean(temp_data[1])
        data_[2,i] = np.mean(temp_data[2])
        data_[3,i] = num_func_eval
        if samples is None:
            data_[4,i] = num_func_eval
        else:
            data_[4,i] = num_func_eval * samples


    save_dict = {'samples': samples, 'exp_val': data_[0], 
                 'para_error': data_[1], 'variance': data_[2], 
                 'func_evals': data_[3], 'facit': facit, 
                 'quantum_evals': data_[4], 'all_data': all_data}

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
    
    if label is 'Multi': color = 'r'
    else: color = 'b'
    
       
    i=0
    for key in data:
        if key == 'exp_val':
            plt.figure(i)
            plt.hlines(data['facit'][0], start, stop, colors='r', 
                       linestyles='dashed', 
                       label='True eig: {}'.format(round(data['facit'][0],4)))
    
            plt.errorbar(data_x,data[key],data['variance'], 
                         fmt='o', label=label, capsize=5, color = color)
            plt.legend()
            plt.xlabel('Function evaluations on the quntum computer')
            plt.ylabel(key)
            i+=1
        elif key == 'para_error':
            plt.figure(i)
            plt.hlines(0, start, stop, colors='r',linestyles='dashed')
            plt.scatter(data_x,data[key],label=label, c=color)
            plt.xlabel('Function evaluations on the quntum computer')
            plt.ylabel(key)
            plt.legend()
            i+=1




################################################################################
# Tests of the methods / Measurments made with the module
################################################################################
def run_bayes_iteration_sweep(j, samples, V=1, i=0, multi = True, ucc = False):
    h = lipkin_quasi_spin.hamiltonian(j,V)[i]

    
    
    if multi is True:
        ansatz_ = ansatz.multi_particle
        qc = ansatz.multi_particle_qc(h)
        file_name = join('bayes_total_evals',
                     'BayesSweep_Samples{}_{}_j{}V{}i{}'.format(samples,ansatz_.__name__,j,V,i))
        ansatz_=ansatz_(h)    
        convert_op = matrix_to_op.multi_particle
        bayes_iteration_sweep(h, qc, ansatz_, convert_op, samples=samples,
                              label='Multi', 
                              save_after_run=True, file_name=file_name)
    if ucc is True:
        ansatz_ = ansatz.one_particle_ucc
        file_name = join('bayes_total_evals',
                     'BayesSweep_Samples{}_{}_j{}V{}i{}'.format(samples,ansatz_.__name__,j,V,i))
        ansatz_=ansatz_(h)    
        qc = ansatz.one_particle_qc
        convert_op = matrix_to_op.one_particle
        bayes_iteration_sweep(h, qc, ansatz_, convert_op, samples=samples,
                              label='One_UCC', 
                              save_after_run=True, file_name=file_name)
    
    

    
    
    





################################################################################
# Main
################################################################################
if __name__ == '__main__':
    for samples in range(1000,2501,500):
        run_bayes_iteration_sweep(1, samples)
    plt.show()

