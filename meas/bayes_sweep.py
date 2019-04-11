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
    
def bayes_iteration_sweep(H, 
                          qc, 
                          ansatz_, 
                          dim_h,
                          samples = None,
                          start = 10, stop = 50, steps = 2, 
                          measurments_per_step=1,
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


    dimension = [(-1.0, 1.0) for i in range(dim_h-1)]


    # Sets up samples, and all vectors to save run in 
    num_evals = range(start, stop, round((stop-start)/steps))
    data_ = np.zeros( (5, len(num_evals)) )
    run_data = []

    # Calculates a facit with sample=None
    facit= vqe_eig.smallest(H, qc, init_params.alternate(dim_h),ansatz_, 
                            disp_run_info=False)
    
    all_data=[]
    for i,num_func_eval in enumerate(num_evals):
        if disp_progress:
            print('Number of function evaluations: {}'.format(num_func_eval))

        temp_data=np.zeros( (3, measurments_per_step) )
        for j in range(measurments_per_step):
            run_data = vqe_eig.smallest_bayes(H, qc, dimension, ansatz_, 
                                              samples, return_all_data=True, 
                                              n_calls= num_func_eval, 
                                              disp=disp_data_during_run)
            temp_data[0,j] = run_data['fun']
            temp_data[1,j] = np.linalg.norm(run_data['x']-facit[1])
            temp_data[2,j] = run_data['expectation_vars']

        all_data.append(temp_data)
        data_[0,i] = np.mean(temp_data[0])
        data_[1,i] = np.mean(temp_data[1])
        data_[2,i] = np.mean(temp_data[2])
        data_[3,i] = num_func_eval
        if samples is None:
            data_[4,i] = num_func_eval * len(H)
        else:
            data_[4,i] = num_func_eval * len(H) * samples

        
        if disp_progress:
            print('Done with calculation: {}/{}'.format(i+1,len(num_evals)))


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
def create_qc(convert_op, h):
    if convert_op is matrix_to_op.one_particle:
        qubit = h.shape[0]
    elif convert_op is matrix_to_op.multi_particle:
        qubit = int.bit_length(h.shape[0])
    else:
        print('Unknown ansatz')
        return
    return get_qc('{}q-qvm'.format(qubit))
    

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

def heatmap_from_data(data_, titel=None, axs=None, interval=None):
    from pandas import DataFrame
    import seaborn as sns
    if axs is None: 
        plt.figure()
        plt.title(titel)

    if interval is not None:
        vmin = interval[0]
        vmax = interval[1]

    if axs is not None: axs.set_title(titel)

    df = DataFrame(data_['para_error'], index=data_['sample_sweep'], 
                   columns=data_['func_eval'])
    

    sns.heatmap(df, ax=axs, square=True, vmin=vmin, vmax=vmax)

def plot_mult_ansatz(file_name_long, matrix_name, directory = None, 
                     interval = None):
    fig, axs = plt.subplots(ncols=3)
    ansatzer = ['multi_particle', 'one_particle', 'one_particle_ucc']

    fig.suptitle(matrix_name)

    for i, axes in enumerate(axs):
        datatitle = join(directory, 
                     str(file_name_long + '_' + ansatzer[i] + '_' + matrix_name + '.pkl'))
        dict_1,_ = data.load(datatitle)
        heatmap_from_data(dict_1, ansatzer[i], axes,
                          interval)
    




################################################################################
# Tests of the methods / Measurments made with the module
################################################################################
def run_bayes_iteration_sweep(ansatz_, convert_op, h=None, j=1, V=1, 
                              samples=None, matrix_num=0, label = None, 
                              save = False, file_name = None, start=10, stop=40,
                              steps=5, measurments=1, plot=False):
    
    if h is None:
        h = lipkin_quasi_spin.hamiltonian(j, V)[matrix_num]

    ans_name = ansatz_.__name__
    ansatz_ = ansatz_(h.shape[0])
    if label is None: label = ans_name
    
    qubit = n_qubits(convert_op,h)

    qc = get_qc('{}q-qvm'.format(qubit))
    H = convert_op(h)
    
    return bayes_iteration_sweep(H, qc, ansatz_, h.shape[0], samples=samples,
                              start=start, stop=stop, steps=steps, 
                              save_after_run=save, label=label, qubits=qubit,
                              ansatz_name=ans_name, file_name=file_name, 
                              measurments_per_step=measurments, plot_after_run=plot)


def heatmap(ansatz_, convert_op, h, label = None, save = False, 
            sample_start=100, sample_stop=1000, sample_step=10, 
            func_start=10, func_stop=40, func_steps=5, file_name=None, 
            plot_after_run=False, measurments=1):
    ansatz_name = ansatz_.__name__
    ansatz_ = ansatz_(h.shape[0])
    
    qc = create_qc(convert_op,h)
    H = convert_op(h)


    samples_sweep = range(sample_start, sample_stop, 
                          round((sample_stop-sample_start)/sample_step))
    func_eval = range(func_start, func_stop, 
                      round((func_stop-func_start)/func_steps))    

    dim = (len(samples_sweep),len(func_eval))
    data_ = np.zeros(dim)
    variance_mesh = np.zeros(dim)
    all_data=[]


    for index,sample in enumerate(samples_sweep):
        temp = bayes_iteration_sweep(H, qc, ansatz_, h.shape[0],
                                     label=label, steps=func_steps, 
                                     start=func_start, 
                                     stop=func_stop, 
                                     samples=sample, 
                                     plot_after_run=False,
                                     measurments_per_step=measurments)
        
        data_[index] = temp['para_error']
        variance_mesh[index] = temp['variance']
        all_data.append(temp['all_data'])
        print('Done with calculation: {}/{}'.format(index+1,len(samples_sweep)))

    save_dict = {'para_error': data_, 'sample_sweep': samples_sweep,
                 'func_eval': func_eval, 'all_runs': all_data,
                 'variance': variance_mesh}

    if save:
        if convert_op is matrix_to_op.one_particle: qubit = h.shape[0]
        elif convert_op is matrix_to_op.multi_particle:
            qubit = int.bit_length(h.shape[0])

        metadata = {'ansatz': ansatz_name, 'Hamiltonian': H,
                    'dimension_of_parameters': '(-1,1)',
                    'minimizer': 'Bayesian Optimizer','Quibits': qubit,
                    'Measurments per average': measurments,
                    'Sample interval': '{}-{}'.format(sample_start,sample_stop),
                    'func_eval interval': '{}-{}'.format(func_start,func_stop),
                    'Type_of_meas': 'Sweep over samples with Bayesian optimizer', 
                    }

        data.save(file = file_name, data=save_dict,metadata=metadata)

    if plot_after_run: heatmap_from_data(save_dict)

def test_parameters(ansatz_, convert_op):
    V = 1
    print('Ansatz: {}'.format(ansatz_.__name__))
    
    for j in range(1,8):
        h_tupple = lipkin_quasi_spin.hamiltonian(j,V)
        for h in reversed(h_tupple):
            if h.shape[0] == 1: continue
            dim_h = h.shape[0]
            temp_ans = ansatz_(dim_h)
            qc = create_qc(convert_op, h)
            H = convert_op(h)
            facit= vqe_eig.smallest(H, qc, init_params.alternate(dim_h),temp_ans, 
                            disp_run_info=False)
            print('Parameter (dim {}): {}'.format(dim_h,facit[1]))





################################################################################
# Main
################################################################################
if __name__ == '__main__':
    '''
    ansatz_ = ansatz.multi_particle
    convert_op = matrix_to_op.multi_particle
    h = lipkin_quasi_spin.hamiltonian(1, 1)[0]
    
    
    heatmap(ansatz_, convert_op, h, save=True,
            sample_step=4, sample_start=100, sample_stop=2000, 
            func_steps=4, func_start=10, func_stop=50, 
            measurments=1, plot_after_run=True)
    plt.show()
    '''
    interval = (0.005,0.12)
    plot_mult_ansatz('updatedSampleDef', 'j1V1i0', 'heatmapsBayes', interval) 
    
    interval = (0.005,0.12)
    plot_mult_ansatz('updatedSampleDef', 'j2V1i1', '1_heatmapsBayes', interval) 
    plt.show()
    


