#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tus Mar 27 14:26 2019

@author: axel
The moduel contains a method that sweeps over samples and perform a Nelder-Mead
iteration for every sample. THe run can be save, and a Help-function exists 
to plot data generated with this module.

The test-cases can be used to run the Main part of the module, or to just plot 
data that has been generated with this module.
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



# Main part of the module
def NM_sample_sweep(H, 
                    qc, 
                    ansatz_, 
                    dim_h,
                    xatol = 1e-3,
                    initial_p = init_params.alternate,
                    start = 1000, stop = 2000, steps = 3, 
                    save_after_run=False, 
                    plot_after_run=True,
                    label = None,
                    ansatz_name=None, 
                    qubits = None,
                    file_name = None):
    """Runs a sweep over a number of samples, defined by start, stop steps. For 
       each given sample performs a Nelder-Mead run with the fatol equal to 
       2 standard deviations. Saves all data from the run and returns it as a
       dictionary. 
    
    Arguments:
        H -- Hamiltonian paulisum
        qc -- quantumcomputer
        ansatz_ -- given ansatz
        dim_h -- dimension of the hamiltonian.
    
    Keyword Arguments:
        xatol -- Step-tolerance for Nelder-Mead (default: {1e-3})
        initial_p -- Inital value for N-M (default: {init_params.alternate})
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
        dict -- Dict with all the data from all the runs.
    """
    
    
    vqe = vqeOverride.VQE_override(minimizer=minimize,
                               minimizer_kwargs={'method': 'Nelder-Mead'})

    # Sets up samples, and all vectors to save run in 
    samples = range(start, stop, round((stop-start)/steps))
    data_ = np.zeros( (5, len(samples)) )
    run_data = []

    # Calculates a facit with sample=None
    facit= vqe_eig.smallest(H, qc, initial_p(dim_h),ansatz_)
    

    for i,sample in enumerate(samples):
        print('Number of samples: {}'.format(sample))
    
        # Makes one calculation with given samples, sets fatol accordingly.
        _, temp_var = vqe.expectation(ansatz_(initial_p(dim_h)), H,
                                        samples=sample,qc=qc)
        fatol_ =2 * np.sqrt(temp_var)
        print('Tolerance: {}'.format(fatol_))

        
        run_data.append( vqe_eig.smallest(H, qc, initial_p(dim_h), ansatz_,
                                          samples=sample, fatol=fatol_, 
                                          xatol=xatol, return_all_data=True) )
        

        data_[0,i], temp_var = vqe.expectation(ansatz_(run_data[-1]['x']), H,
                                        samples=sample,qc=qc)
        
        data_[1,i] = np.linalg.norm(run_data[-1]['x']-facit[1])
        data_[2,i] = 2*np.sqrt(temp_var)
        data_[3,i] = len(run_data[-1]['expectation_vals'])
        data_[4,i] = len(H)*sample
        
        print('Done with calculation: {}/{}'.format(i+1,len(samples)))


    save_dict = {'samples': samples, 'exp_val': data_[0], 'facit': facit,
                 'para_error': data_[1], 'variance': data_[2], 
                 'iterations': data_[3], 'func_evals': data_[4], 
                 'all_data': run_data}

    if save_after_run:
        metadata = {'ansatz': ansatz_name, 'Hamiltonian': H, 'xatol': xatol, 
                    'minimizer': 'Nelder-Mead','Qubits': qubits,
                    'Type_of_meas': 'eig, var, para errors and iterations', 
                    'fatol': 'Varierande med variansen'}

        data.save(file = file_name, data=save_dict, metadata=metadata)

    if plot_after_run: 
        if label is None: label = ansatz_name
        plot_run(save_dict, label=label)
    
    return save_dict
    

def NM_sample_sweep_zeros(H, 
                    qc, 
                    ansatz_, 
                    dim_h,
                    xatol = 1e-3,
                    initial_p = init_params.alternate,
                    start = 1000, stop = 2000, steps = 3, 
                    save_after_run=False, 
                    plot_after_run=True,
                    label = None,
                    ansatz_name=None, 
                    qubits = None,
                    file_name = None):
    """Runs a sweep over a number of samples, defined by start, stop steps. For 
       each given sample performs a Nelder-Mead run with the fatol equal to 
       2 standard deviations. Saves all data from the run and returns it as a
       dictionary. 
    
    Arguments:
        H -- Hamiltonian paulisum
        qc -- quantumcomputer
        ansatz_ -- given ansatz
        dim_h -- dimension of the hamiltonian.
    
    Keyword Arguments:
        xatol -- Step-tolerance for Nelder-Mead (default: {1e-3})
        initial_p -- Inital value for N-M (default: {init_params.alternate})
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
        dict -- Dict with all the data from all the runs.
    """
    print(H)
    
    vqe = vqeOverride.VQE_override(minimizer=minimize,
                               minimizer_kwargs={'method': 'Nelder-Mead'})

    # Sets up samples, and all vectors to save run in 
    samples = range(start, stop, round((stop-start)/steps))
    data_ = np.zeros( (3, len(samples)))
    

    # Calculates a facit with sample=None
    facit= vqe_eig.smallest(H, qc, initial_p(dim_h),ansatz_)
    

    for i,sample in enumerate(samples):
        print('Number of samples: {}'.format(sample))
    
        # Makes one calculation with given samples, sets fatol accordingly.
        _, temp_var = vqe.expectation(ansatz_(initial_p(dim_h)), H,
                                        samples=sample,qc=qc)
        
        inital = np.zeros(dim_h-1)

        data_[0,i], temp_var = vqe.expectation(ansatz_(inital), H,
                                        samples=sample,qc=qc)
        
        data_[1,i] = 2*np.sqrt(temp_var)
        data_[2,i] = sample*len(H)     
        print('Done with calculation: {}/{}'.format(i+1,len(samples)))


    save_dict = {'func_evals': data_[2], 'exp_val': data_[0],
                 'variance': data_[1], 'facit': facit}

    if save_after_run:
        metadata = {'ansatz': ansatz_name, 'Hamiltonian': H, 'xatol': xatol, 
                    'minimizer': 'Nelder-Mead','Qubits': qubits,
                    'Type_of_meas': 'eig, var, para errors and iterations', 
                    'fatol': 'Varierande med variansen'}

        data.save(file = file_name, data=save_dict, metadata=metadata)

    if plot_after_run: 
        if label is None: label = ansatz_name
        plot_run(save_dict, label=label)
    
    return save_dict


################################################################################
# Extra methods
################################################################################

# Help method that plots a run
def plot_run(data, meta=None, label=None ):
    """Plot data from a run of the method above.
    Arguments:
        data {dict} -- Dictionary generated with the method above. 
    
    Keyword Arguments:
        label {srt} -- Label for the legend (default: {None})
    """
    if data.__contains__('func_eval') is False:
        if meta is None: 
            print('Missing metadata')
            return
        data['func_evals'] = [data*len(meta['Hamiltonian']) 
                             for data in data['samples']]




    start = data['func_evals'][0]
    stop = data['func_evals'][-1]
    
    if label is None: label =  'Check metadata for ansatz'   
    i=0
    for key in data:
        if key == 'facit' or key == 'samples' or key == 'all_data': continue
        elif key == 'exp_val':
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
        else:
            plt.figure(i)
            if key == 'para_error':         
                plt.hlines(0, start, stop, colors='r',linestyles='dashed')
            plt.scatter(data['func_evals'],data[key],label=label)
            plt.xlabel('Function evaluations')
            plt.ylabel(key)
            plt.legend()
            i+=1



################################################################################
# Tests of the methods / Measurments made with the module
################################################################################
def run_NM_sample_sweep(ansatz_, convert_op, h=None, j=1, V=1, matrix_num=0, 
                        label = None, save = False, plot = True, 
                        file_name = None,
                        start=100, stop=8000, steps=50):
    if h is None:
        h = lipkin_quasi_spin.hamiltonian(j, V)[matrix_num]

    ans_name = ansatz_.__name__
    if ans_name == 'one_particle_ucc': inital_p = init_params.ones
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
    
    return NM_sample_sweep_zeros(H, qc, ansatz_, h.shape[0], start=start, stop=stop, 
                           steps=steps, save_after_run=save, label=label, 
                           ansatz_name=ans_name, qubits=qubit, 
                           file_name=file_name, plot_after_run=plot)


def plot_from_data():
    datatitle = join('over_night_run', 'OverNightRun3.pkl')
    dict_1, meta = data.load(datatitle)
    
    plot_run(dict_1, label=meta['ansatz'])
    datatitle = join('over_night_run', 'OverNightRun4.pkl')
    dict_1, meta = data.load(datatitle)
    
    plot_run(dict_1, label=meta['ansatz'])
    datatitle = join('over_night_run', 'OverNightRun5.pkl')
    dict_1, meta = data.load(datatitle)
    
    plot_run(dict_1, label=meta['ansatz'])

    


def multiple_ansatzer(j=1, V=1, matrix_num=0):
    h_ = lipkin_quasi_spin.hamiltonian(j, V)[matrix_num]


    ansatz_ = ansatz.one_particle
    convert_op = matrix_to_op.one_particle
    #run_NM_sample_sweep(ansatz_, convert_op, h=h_, label='One')


    ansatz_ = ansatz.one_particle_ucc
    run_NM_sample_sweep(ansatz_, convert_op, h=h_, label='One UCC')


    ansatz_ = ansatz.multi_particle
    convert_op = matrix_to_op.multi_particle
    run_NM_sample_sweep(ansatz_, convert_op, h=h_, label='Multi')

def multiple_runs_and_save(h, count):
    ansatzer = [ansatz.one_particle, ansatz.one_particle_ucc, 
                ansatz.multi_particle]
    
    for index, ansatz_ in enumerate(ansatzer):
        file_name = join('over_night_run', 'OverNightRun{}'.format(count))
        convert_op = matrix_to_op.one_particle

        if index == 2: convert_op = matrix_to_op.multi_particle
        run_NM_sample_sweep(ansatz_, convert_op, h, save=True, 
                            file_name=file_name,
                            start=100, stop=10000, steps=100)
        count+=1
    
    return count
    
def print_H():
    
    print(matrix_to_op.one_particle(h))
    print(matrix_to_op.multi_particle(h))

    

################################################################################
# Main
################################################################################
if __name__ == '__main__':
    plot_from_data()
    plt.show()

