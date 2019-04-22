"""
Recipe for long runs. This makes sure to save often and keeps the data in a
single file.

@author = Joel
"""
from core.data import save, load
from os.path import join, basename
import os
from multiprocessing import Pool
from core import lipkin_quasi_spin


# TODO: (for Joel) check if pauli-objects are thread-safe (they probably are)
#  write new save style and append (with open(file, 'ab')) in data (using
#       multiple objects in pickle)
#  write example input_iterate() and simulate()
#  write class that takes iterator and set and wraps it to iterate through
#       the iterator but only yield the second output if the first is not in
#       the set (in that case, add the first to the set and save to metafile)
#  cache output_func


# TODO: When writing a meas script, change (only) the parts marked by TODOs.

# TODO: give a version-number of the script (this should be changed iff the
#  meaning of the elements in the tuple yielded by the generator (
#  identifier_generator) changes. Changing this will start a completely new
#  simulation (the default behaviour of this script is to continue where it
#  stopped last time it was run). The version-number will be added to the
#  file name (e.g test_v1_...)
version = 1

# TODO: select directory and file to save to (the case is appended to the file).
directory = 'test_recipe'  # directory to save to
file = 'test'  # file to save to

# Append version number to file
file += f'_v{version}'


# Metadata about what has been done previously will be saved here:
path_metadata = join(directory, file + '_metadata')

# Load/initialize metadata
try:
    # Try to load the file (will raise FileNotFoundError if not existing)
    metadata = load(path_metadata)[0]
except FileNotFoundError:
    metadata = {'identifiers': {}}
    metametadata = {'description': "File that keeps track of what's been done "
                                   "previously in this script "
                                   f"({basename(__file__)})."}
    save(file=path_metadata, data=metadata, metadata=metametadata)


# TODO: identifiers to iterate over, should yield a tuple that uniquely
#  identifies the simulation (using floats/ints as elements is a good idea),
#  e.g (1, 2, 3, 4)
def identifier_generator():
    V = 1.
    for j in range(6):
        for i in range(2):
            for samples in range(100, 1000, 100):
                for num_sim in range(5):
                    yield (V, j, i, samples, num_sim)


# TODO
def input_0():
    # could return e.g ansatz
    return [None]


# TODO
def input_3(V, j, i):
    return [lipkin_quasi_spin.hamiltonian(V, j)[i]]


# TODO: the function that runs e.g. smallest. The inputs to this function are
#  created by an iterator (input_iterate below). Make sure to create
#  non-thread/multiprocess-safe objects (such as VQE-objects) here rather than
#  in the iterator.

# TODO: the arguments to simulate corresponds to:
#  first_element_from_identifier_generator, second_element_from_id_gen, ...
#  first_element_from_first_input_func, second_element_from_first_input_func, ..
#  ...
#  first_element_from_last_input_func, second_element_from_last_input_func

def simulate(V, j, i, samples, dummy, h):
    # TODO: create VQE-object here! (not multiprocess safe)
    result = {'x': j+i+samples, 'fun': j*i*samples}
    return result


class Bookkeeper:
    """
    Class for keeping track of what's been done and only assign new tasks.
    """
    def __init__(self, iterator, book, callback=lambda: None,
                 output_calc=None):
        if output_calc is None:
            output_calc = {}
        self.iterator = iterator
        self.book = book
        self.callback = callback
        self.output_calc = output_calc

    def __iter__(self):
        return self

    def __next__(self):
        x = self.iterator.__next__()
        if x not in self.book or self.book[x] is not True:
            self.book[x] = True
            self.callback()
            output = []
            for i in range(len(x)+1):
                if i in self.output_calc:
                    output.extend(self.output_calc[i](
                        *[y for j, y in enumerate(x) if j < i]))
            yield x, output


def wrap(x):
    return x[0], simulate(*x[0], *x[1])


# Might want to change these to improve performance
num_workers = os.cpu_count()
max_task = 1
chunksize = 1

generator = Bookkeeper(identifier_generator, metadata['identifiers'],
                       lambda: quick_save(path_metadata, metadata,
                                          metametadata),
                       {0: input_0,
                        3: input_3})

with Pool(num_workers, maxtasksperchild=max_task) as p:
    result_generator = p.imap_unordered(wrap, generator, chunksize=chunksize)
    for identifier, result in result_generator:
        # Use identifier and result to determine file and append to that file
        # using data.append

        # Note that the results will be unordered so make sure to save enough
        # info!
        pass
