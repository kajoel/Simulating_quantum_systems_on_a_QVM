"""
Recipe for long runs. This makes sure to save often and keeps the data in a
single file.

@author = Joel, Carl
"""
import core.interface
from core import data
from os.path import join, basename, isfile
import os
from multiprocessing import Pool
from core import lipkin_quasi_spin
from functools import lru_cache
from core.interface import hamiltonians_of_size, vqe_nelder_mead
from core import matrix_to_op
from core import ansatz
from core import vqe_eig
from core import callback as cb
import numpy as np
import sys
import warnings
from constants import ROOT_DIR
from uuid import getnode as get_mac
from time import perf_counter

# TODO: When writing a meas script, change (only) the parts marked by TODOs.
#  MAKE SURE TO SAFE ENOUGH INFORMATION!
#  BETTER TO SAVE TOO MUCH THAN TOO LITTLE!

# Input number of workers
max_num_workers = os.cpu_count()
if len(sys.argv) <= 1:
    num_workers = max_num_workers
else:
    try:
        num_workers = int(sys.argv[1])
    except ValueError:
        num_workers = max_num_workers
        warnings.warn(f'Could not parse input parameter 1 num_workers.')

# Input start of range
if len(sys.argv) <= 2:
    start_range = 0
else:
    try:
        start_range = int(sys.argv[2])
    except ValueError:
        start_range = 0
        warnings.warn(f'Could not parse input parameter 2 start_range.')

# Input end of range
if len(sys.argv) <= 3:
    stop_range = np.inf
else:
    try:
        stop_range = int(sys.argv[3])
    except ValueError:
        stop_range = np.inf
        warnings.warn(f'Could not parse input parameter 3 stop_range.')

print(f'\nStarting with num_workers = {num_workers}, and range = '
      f'[{start_range}, {stop_range})\n')

# TODO: give a version-number of the script (this should be changed iff the
#  meaning of the elements in the tuple yielded by the generator (
#  identifier_generator) changes. Changing this will start a completely new
#  simulation (the default behaviour of this script is to continue where it
#  stopped last time it was run). The version-number will be added to the
#  file name (e.g test_v1_...)
version = 3

# TODO: select directory and basename of file to save to.
directory = 'NM_Restart_Parallel'  # directory to save to
file = ''  # file to save to (basename)

# Base dir, save to gitignored directory to avoid problems
base_dir = join(ROOT_DIR, 'data_ignore')

# Append version number to file
file += f'v{version}'
directory += f'_v{version}'

# Make subdirectory based on MAC-address (to allow for multiple computers)
directory = join(directory, str(get_mac()))

# Metadata about what has been done previously will be saved here:
path_metadata = join(directory, file + '_metadata')

# Load/initialize metadata
try:
    # Try to load the file (will raise FileNotFoundError if not existing)
    metadata, metametadata = data.load(path_metadata, base_dir=base_dir)
except FileNotFoundError:
    metadata = []
    metametadata = {'description': "File that keeps track of what's been done "
                                   "previously in this script "
                                   f"({basename(__file__)}).",
                    'run_time': [],
                    'num_workers': [],
                    'num_tasks_completed': [],
                    'success_rate': []}
    data.save(file=path_metadata, data=metadata, metadata=metametadata,
              extract=True, base_dir=base_dir)

# Extract identifiers to previously completed simulations
ids = set()
for id_, value in metadata:
    if value is True:
        if id_ in ids:
            raise KeyError('Multiple entries for same id in metadata.')
        ids.add(id_)

# Cleanup (metadata could contain hundreds of Exception objects)
del metadata, metametadata


# TODO: identifiers to iterate over, should yield a tuple that uniquely
#  identifies the simulation (using floats/ints as elements is a good idea,
#  e.g (1, 2, 3, 4)), i.e. every call to smallest, vqe.run or similar should
#  have a unique identifier.
def identifier_generator():
    # ansatz
    ansatz_name = 'multi_particle'
    # size of hamiltonian
    for size in range(2, 3):
        # the index of the four hamiltonians
        for hamiltonian_idx in range(4):
            # number of samples
            for samples in np.linspace(400, 60000, 100):
                # input_4 is effectively called here with four arguments
                for max_same_para in range(1, 10):
                    for repeats in range(5):
                        # input_5 is effectively called here with five arguments
                        yield (ansatz_name, size, hamiltonian_idx,
                               int(round(samples)), max_same_para, repeats)


# TODO: Functions for creating objects (things larger than ints/floats) that
#  simulate (below) can use. This is a good place to create e.g. the
#  hamiltonian. input_n will get the first n items in the identifiers. Try to
#  setup things as early as possible (better to do it in input_0 than input_1
#  and so on) to avoid creating identical objects (and save some time). These
#  have to return an object that can be iterated over (like a tuple or list).
#  Make sure to include @lru_cache(maxsize=1) on the line above def.
#  You do not have to define all (or any) of input_0, input_1, ...
#  It's a also a good idea to print things like "starting on j=3" here.

@lru_cache(maxsize=1)
def input_3(ansatz_name, size, hamiltonian_idx):
    h, eig = hamiltonians_of_size(size)
    return h[hamiltonian_idx], eig[hamiltonian_idx]


@lru_cache(maxsize=1)
def input_5(ansatz_name, size, hamiltonian_idx, samples, max_same_para):
    print(f'Size={size}, Hamiltonian_idx={hamiltonian_idx}, '
          f'Samples={samples}, Max_same_para={max_same_para}')
    return ()


# TODO: add your defined input functions to the dictionary below. The key is
#  how many elements from the identifier the input function will be called
#  with.
input_functions = {3: input_3,
                   5: input_5}


# TODO: define constants needed in simulate (below). If you change any of
#  these you must change the version (above) to not sa


# TODO: the function that runs e.g. smallest or vqe.run. Make sure to create
#  non-multiprocess-safe objects (such as VQE-objects) here rather than
#  in the input_n functions. Let id be an identifier; the arguments to this
#  function will then be:
#  id[0], id[1], ..., input_0()[0], input_0()[1], ..., input_1(id[0])[0],
#  input_1(id[0])[1], ..., ..., input_N(id[0], id[1], ..., id[N-1])[0],
#  input_N(id[0], id[1], ..., id[N-1])[1], ...
#  (only including the defined input functions)
def simulate(ansatz_name, size, hamiltonian_idx, samples, max_same_para,
             repeats, h, eig):
    # TODO: create VQE-object here! (not multiprocess safe)
    # TODO: run e.g. smallest here and return result.
    H, qc, ansatz_, initial_params = core.interface.create_and_convert(ansatz_name, h)
    vqe = vqe_nelder_mead(samples=samples, H=H)
    tol_para = 1e-3
    callback = cb.restart_break(max_same_para, tol_para)
    max_fun_evals = 100
    result = vqe_eig.smallest(H, qc, initial_params, vqe,
                              ansatz_, samples,
                              callback=callback, max_fun_evals=max_fun_evals)
    result.correct = eig
    return result


# TODO: function that takes in identifier and outputs the file (as a string)
#  to save to. Keep the number of files down!
def file_from_id(identifier):
    return file + f'_{identifier[0]}_size={identifier[1]}_matidx'\
                  f'={identifier[2]}'


# TODO: function that takes in identifier and outputs metadata-string.
def metadata_from_id(identifier):
    return {'description': 'Nelder-Mead Restart, identifier = data[:][0], '
                           'result = data[:][1]',
            'size': identifier[1],
            'matidx': identifier[2],
            'ansatz': identifier[0]}


class Bookkeeper:
    """
    Class for keeping track of what's been done and only assign new tasks.
    """

    def __init__(self, iterator, book, output_calc=None, bounds=None):
        """

        :param iterator: Iterable
        :param set book: Set of identifiers corresponding to previously
            completed tasks.
        :param output_calc: List of functions
        :param bounds: Bounds of elements to return from iterator
        """
        if output_calc is None:
            output_calc = {}
        if bounds is None:
            bounds = [0, np.inf]
        self.iterator = iterator
        self.book = book
        self.output_calc = output_calc
        self.bounds = bounds
        self.count = -1  # to start from 0 (see += 1 below)

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            x = self.iterator.__next__()
            self.count += 1

            if self.count >= self.bounds[1]:
                raise StopIteration

            if x not in self.book and self.count >= self.bounds[0]:
                output = []
                for i in range(len(x) + 1):
                    if i in self.output_calc:
                        output.extend(self.output_calc[i](
                            *[y for j, y in enumerate(x) if j < i]))
                return x, output


def wrap(x):
    # Use a broad try-except to not crash if we don't have to
    try:
        return x[0], simulate(*x[0], *x[1])

    except Exception as e:
        # This will be saved in the metadata file so we can check success-rate.
        return x[0], e


# Might want to change these to improve performance
max_task = 1
chunksize = 1

generator = Bookkeeper(identifier_generator(), ids, input_functions,
                       [start_range, stop_range])
files = set()
success = 0
fail = 0
start_time = perf_counter()

try:
    with Pool(num_workers, maxtasksperchild=max_task) as p:
        result_generator = p.imap_unordered(wrap, generator,
                                            chunksize=chunksize)
        for identifier, result in result_generator:
            # Handle exceptions:
            if isinstance(result, Exception):
                fail += 1
                # Save the error
                data.append(path_metadata, [identifier, result],
                            base_dir=base_dir)
            else:
                success += 1
                file_ = file_from_id(identifier)
                if file_ not in files:
                    files.add(file_)
                    if not isfile(join(base_dir, directory,
                                       file_ + '.pkl')):
                        # Create file
                        metadata = metadata_from_id(identifier)
                        data.save(join(directory, file_), [], metadata,
                                  extract=True, base_dir=base_dir)

                # TODO: Save results and identifier
                #  Note that the results will be unordered so make sure to save
                #  enough info!
                data.append(join(directory, file_), [identifier, result],
                            base_dir=base_dir)

                # Mark the task as completed (last in the else,
                # after saving result)
                data.append(path_metadata, [identifier, True],
                            base_dir=base_dir)
finally:
    stop_time = perf_counter()
    if success + fail == 0:
        fail = 1  # To avoid division by zero

    # Post simulation.
    metadata, metametadata = data.load(path_metadata, base_dir=base_dir)
    meta_dict = {}
    for x in metadata:
        # Keep only the last exception for given identifier.
        if x[0] not in meta_dict or meta_dict[x[0]] is not True:
            meta_dict[x[0]] = x[1]
    metadata = [[x, meta_dict[x]] for x in meta_dict]
    metametadata['run_time'].append(stop_time - start_time)
    metametadata['num_workers'].append((num_workers, max_num_workers))
    metametadata['num_tasks_completed'].append(success)
    metametadata['success_rate'].append(success / (success + fail))
    data.save(file=path_metadata, data=metadata, metadata=metametadata,
              extract=True, base_dir=base_dir)

    # Print some stats
    print('\nSimulation completed.')
    print(f'Total number of tasks this far: {len(metadata)}')
    print(f'Completed tasks this run: {success}')
    print(f'Success rate this run: {success / (success + fail)}')
    done = sum(x[1] for x in metadata if x[1] is True)
    print(f'Number of tasks remaining: {len(metadata) - done}')