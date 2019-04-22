"""
@author = Sebastian, Carl
"""
import numpy as np
from pyquil import get_qc

from core import matrix_to_op, ansatz, init_params, create_vqe, vqe_eig
from core.data import save, load
from os.path import join
from constants import ROOT_DIR
import os
import sys
from multiprocessing import Pool
import itertools
import warnings
from time import perf_counter
from core import callback as cb

from core.lipkin_quasi_spin import hamiltonian, eigs

###################### FUNCTIONS ######################

def fatol_skattning(sample, H):
    return sum(np.abs(term.coefficient) for term in H.terms) / np.sqrt(sample)

######################################################

if len(sys.argv) <= 1:
    case = 0
else:
    try:
        case = int(sys.argv[1])
    except ValueError:
        case = 0
        warnings.warn('Could not parse input parameter {}. Using case=0'.format(
            sys.argv[1]))

# Number of times each simulation is run, can be redefined in case below
num_sim = 10

if case == 0:
    ansatz_name = 'multi_particle'
elif case == 1:
    ansatz_name = 'one_particle_ucc'
elif case == 2:
    ansatz_name = 'one_particle'
elif case == 3:
    ansatz_name = 'multi_particle_ucc'
else:
    raise ValueError('The case-defining input is to large.')


directory = 'NM_Restart_Parallel'  # directory to save to
file = 'NM_Restart_test'  # file to save to

# Complete path to the saved file (relative to the data directory):
path = join(directory, file + '_' + str(case))


try:
    # Try to load the file (will raise FileNotFoundError if not existing)
    data, metadata = load(path)
except FileNotFoundError:
    # TODO: Initialize data and metadata. Write description and add more
    #  metadata fields.
    data = {'x': [],
            'y': [],
            'z': [],
            'result': []}
    metadata = {'description': 'my description',
                'time': [],  # Save times for analysis (modify if you wish)
                'count': 0}  # IMPORTANT! Don't touch!

V, matrix = 1, 0
xatol = 1e-2
tol_para = 1e-2
max_iter = 20
samples = 2000
max_para = 5


# TODO: the function that runs smallest. The inputs to this function is
#  iterated over while constant parameters should be defined in cases above.

def simulate(j):
    print("j = %i" % j)
    h = hamiltonian(j, V)[matrix]
    print(h.todense())
    dim = h.shape[0]
    H, qc, ansatz_, initial_params = ansatz.create(ansatz_name, h, dim)
    vqe = create_vqe.default_nelder_mead()
    callback = cb.restart_on_same_param(max_para, tol_para, True)
    exp_val = vqe_eig.smallest(H, qc, initial_params, vqe, ansatz_, samples,
                              disp=True, callback=callback)

    return exp_val


# TODO: input parameters to iterate over, should yield tuples.
def input_iterate(case):
    # With the example simulate (above) this iterate is equivalent to
    # for x in range(1):'
    #     for y in range(2):
    #         for z in [1, 10]:
    #             result = simulate(x, y, z)
    return itertools.product([1, 2])


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
    for x in input_iterate(case):
        count += 1
        if count > metadata['count']:
            # In this case the relevant simulation has not already been saved
            # to the file and is therefore run now.

            # Run a few times in parallel using multiprocess.Pool and map.
            # result will be a list (of length num_sim) containing what simulate
            # returns. If simulate returns e.g. x, y the elements in the list
            # are tuples (x, y).
            start_time = perf_counter()  # TODO: remove if you don't want time
            result = p.map(wrap, run(num_sim, x))
            stop_time = perf_counter()  # TODO: remove if you don't want time

            # TODO: add the results to the data object
            data['x'].append(x[0])
            data['y'].append(x[1])
            data['z'].append(x[2])
            data['result'].append(result)

            # Update metadata['count'] and save file:
            metadata['count'] = count
            # TODO: remove (the line below) if you don't want time (time in s)
            metadata['time'].append(stop_time - start_time)
            save(path, data, metadata)
