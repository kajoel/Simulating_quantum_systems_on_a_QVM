"""
Recipe for long runs. This makes sure to save often and keeps the data in a
single file.

@author = Joel
"""
from core.data import save, load
from os.path import join
import os
import sys
from multiprocessing import Pool
import itertools
import warnings
from time import perf_counter
import numpy as np


# TODO: When writing a meas script, change (only) the parts marked by TODOs.

if len(sys.argv) <= 1:
    case = 0
else:
    try:
        case = int(sys.argv[1])
    except ValueError:
        case = 0
        warnings.warn('Could not parse input parameters. Using case=0')

# Number of times each simulation is run, can be redefined in case below
num_sim = 1

# TODO: Set upp a few different cases that defines the run.
#  0 should be a light test case. These cases could have different minimizers,
#  ansätze, etc.. (These are relatively constant; if you wan't to loop over a
#  parameter, use input_iterate below instead.)
if case == 0:
    pass
elif case == 1:
    num_sim = 5
elif case == 2:
    num_sim = 100
else:
    raise ValueError('The case-defining input is to large.')

# TODO: select directory and file to save to (the case is appended to the file).
directory = 'test_recipe'  # directory to save to
file = 'test'  # file to save to

# Complete path to the saved file (relative to the data directory):
path = join(directory, file + '_' + str(case))

try:
    # Try to load the file (will raise FileNotFoundError if not existing)
    data, metadata = load(path)
except FileNotFoundError:
    # TODO: Initialize data and metadata. Write description and add more
    #  metadata fields.
    data = {'j': [],
            'i': [],
            'samples': [],
            'x': [],
            'fun': []}
    metadata = {'description': 'my description',
                'time': [],  # Save times for analysis (modify if you wish)
                'count': 0}  # IMPORTANT! Don't touch!


# TODO: the function that runs smallest. The inputs to this function is
#  iterated over while constant parameters should be defined in cases above.
def simulate(j, i, samples):
    # TODO: create VQE-object here! (not multiprocess safe)
    result = {'x': j+i+samples, 'fun': j*i*samples}
    return result


# TODO: input parameters to iterate over, should yield tuples.
def input_iterate(case):
    # With the example simulate (above) this iterate is equivalent to
    # for j in range(3):
    #     for i in range(2):
    #         for samples in [1, 10]:
    #             result = simulate(j, i, samples)

    # Another example:
    # inputs = []
    # for j in range(3):
    #     for i in range(j):  # range of i depending of j
    #         for samples in [1, 10]:
    #             inputs.append((j, i, samples))
    # return  inputs
    return itertools.product(range(3), range(2), [1, 10])


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
            results = p.map(wrap, run(num_sim, inputs))
            stop_time = perf_counter()  # TODO: remove if you don't want time

            # TODO: add the results to the data object (might want to save
            #  both mean and variance or possibly even every data point)
            data['j'].append(inputs[0])
            data['i'].append(inputs[1])
            data['samples'].append(inputs[2])
            data['x'].append(np.array([result['x'] for result in results]))
            data['fun'].append(np.array([result['fun'] for result in results]))

            # Update metadata['count'] and save file:
            metadata['count'] = count
            # TODO: remove (the line below) if you don't want time (time in s)
            metadata['time'].append(stop_time-start_time)
            save(path, data, metadata)
