"""
Long run of bayesin optimization that uses parallel computations. Uses the 
one_particle_ucc ansatz

@author: Recipe: Joel, filled in TODO: Axel
"""
from core.data import save, load
from os.path import join
import os
import sys
from multiprocessing import Pool
import itertools
import warnings
from time import perf_counter
from core import ansatz, vqe_eig, vqe_override, matrix_to_op, create_vqe, \
                 lipkin_quasi_spin, init_params
import numpy as np


if len(sys.argv) <= 1:
    case = 0
else:
    try:
        case = int(sys.argv[1])
    except ValueError:
        case = 0
        warnings.warn('Could not parse input parameters. Using case=0')

# Number of times each simulation is run, can be redefined in case below
num_sim = 5


ansatz_ = ansatz.one_particle_ucc
convert_op = matrix_to_op.one_particle

if case == 0:
    h = lipkin_quasi_spin.hamiltonian(1, 1)[0]
    
elif case == 1:
    h = lipkin_quasi_spin.hamiltonian(2, 1)[0]
    
elif case == 2:
    h = lipkin_quasi_spin.hamiltonian(2, 1)[1]
    
elif case == 3:
    h = lipkin_quasi_spin.hamiltonian(3, 1)[0]
    
elif case == 4:
    h = lipkin_quasi_spin.hamiltonian(3, 1)[1]
    
elif case == 5:
    h = lipkin_quasi_spin.hamiltonian(4, 1)[0]
    
elif case == 6:
    h = lipkin_quasi_spin.hamiltonian(4, 1)[1]
    
elif case == 7:
    h = lipkin_quasi_spin.hamiltonian(5, 1)[0]
    
elif case == 8:
    h = lipkin_quasi_spin.hamiltonian(5, 1)[1]
    
elif case == 9:
    h = lipkin_quasi_spin.hamiltonian(6, 1)[0]
    
elif case == 10:
    h = lipkin_quasi_spin.hamiltonian(6, 1)[1]
    
else:
    raise ValueError('The case-defining input is to large.')

# Select directory and file to save to (the case is appended to the file).
directory = 'bayes_total_evals'  # directory to save to
file = 'bayes_parallel_one_particle_ucc_updated'  # file to save to

# Complete path to the saved file (relative to the data directory):
path = join(directory, file + '_' + str(case))

try:
    # Try to load the file (will raise FileNotFoundError if not existing)
    data, metadata = load(path)
except FileNotFoundError:
    # Initialize data and metadata. Write description and add more
    #  metadata fields.
    data = {'para_error': [],
            'variance': [],
            'n_calls': [],
            'samples': [],
            'result': [], 
            'parameters': []}

    metadata = {'description': 'Sweep over func evals with Bayesian optimizer. \
                Cases are iterations over the value of j, from 1 to 6.',
                'ansatz': ansatz_.__name__,
                'minimizer': 'Bayesian Optimizer', 'num_sim': num_sim,
                'time': [],  # Save times for analysis (modify if you wish)
                'count': 0}  # IMPORTANT! Don't touch!


# The function that runs smallest. The inputs to this function is
#  iterated over while constant parameters should be defined in cases above.
def simulate(n_calls, samples):
    dimension = [(-5.0, 5.0)]*(h.shape[0]-1)
    temp_ansatz = ansatz_(h)
    qc = ansatz.one_particle_qc(h)
    H = convert_op(h)

    vqe_nm = create_vqe.default_nelder_mead()
    facit = vqe_eig.smallest(H, qc, init_params.ucc(h.shape[0]), vqe_nm,
                             temp_ansatz, disp=False)
   
    vqe = create_vqe.default_bayes(n_calls=n_calls)
    
    result = np.zeros(2)
    run_data = vqe_eig.smallest(H, qc, dimension, vqe, temp_ansatz, 
                                samples=samples, disp=False)
    
    result[0] = np.linalg.norm(run_data['x']-facit['x'])
    result[1] = np.mean(run_data['expectation_vars'])
    parameter = run_data['x']

    return result, parameter


# Input parameters to iterate over, should yield tuples.
def input_iterate(case):
    # With the example simulate (above) this iterate is equivalent to
    # for x in range(1):
    #     for y in range(2):
    #         for z in range([1, 10]):
    #             result = simulate(x, y, z)
    inputs = []
    num_para = h.shape[0] -1

    for n_calls in range(10, 30 + num_para*15, 5 + 5*(num_para-1)):
        for samples in range(100, 250 + 2000*num_para, 250*num_para):
            inputs.append((n_calls, samples))

    return inputs


# Wrap simulate to unpack the arguments from iterate
def wrap(x):
    return simulate(*x)


# Wrapper to run multiple simulations with same inputs.
def run(num, x):
    for i in range(num):
        yield x


count = 0  # keep track of what's been done
num_workers = min(num_sim, os.cpu_count())
with Pool(num_workers, maxtasksperchild=1) as p:
    for inputs in input_iterate(case):
        count += 1
        if count > metadata['count']:
            # In this case the relevant simulation has not already been saved
            # to the file and is therefore run now.

            # Run a few times in parallel using multiprocess.Pool and map.
            # result will be a list (of length num_sim) containing what simulate
            # returns. If simulate returns e.g. x, y the elements in the list
            # are tuples (x, y).
            start_time = perf_counter()  # TODO: remove if you don't want time
            result = p.map(wrap, run(num_sim, inputs))
            stop_time = perf_counter()  # TODO: remove if you don't want time

            # TODO: add the results to the data object
            data['para_error'].extend(result[i][0][0] for i in range(len(result)))
            data['variance'].extend(result[i][0][1] for i in range(len(result)))
            data['n_calls'].extend([inputs[0]]*num_sim)
            data['samples'].extend([inputs[1]]*num_sim)
            data['parameters'].extend(result[i][1] for i in range(len(result)))

            data['result'].append(result)

            # Update metadata['count'] and save file:
            metadata['count'] = count
            # TODO: remove (the line below) if you don't want time (time in s)
            metadata['time'].append(stop_time - start_time)
            save(path, data, metadata)
