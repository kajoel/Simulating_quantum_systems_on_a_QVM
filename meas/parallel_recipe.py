"""
Recipe for long runs. This makes sure to save often and keeps the data in a
single file.

@author = Joel
"""
from core import data
from os.path import join, basename, isfile
import os
from multiprocessing import Pool
from core import lipkin_quasi_spin
from functools import lru_cache


# TODO: (for Joel) check if pauli-objects are thread-safe (they probably are)
#  check if qc is thread-safe


# TODO: When writing a meas script, change (only) the parts marked by TODOs.
#  MAKE SURE TO SAFE ENOUGH INFORMATION!
#  BETTER TO SAVE TOO MUCH THAN TOO LITTLE!

# TODO: give a version-number of the script (this should be changed iff the
#  meaning of the elements in the tuple yielded by the generator (
#  identifier_generator) changes. Changing this will start a completely new
#  simulation (the default behaviour of this script is to continue where it
#  stopped last time it was run). The version-number will be added to the
#  file name (e.g test_v1_...)
version = 1

# TODO: select directory and basename of file to save to.
directory = 'test_recipe'  # directory to save to
file = 'test'  # file to save to (basename)

# Append version number to file
file += f'_v{version}'


# Metadata about what has been done previously will be saved here:
path_metadata = join(directory, file + '_metadata')

# Load/initialize metadata
try:
    # Try to load the file (will raise FileNotFoundError if not existing)
    metadata, metametadata = data.load(path_metadata)
except FileNotFoundError:
    metadata = []
    metametadata = {'description': "File that keeps track of what's been done "
                                   "previously in this script "
                                   f"({basename(__file__)})."}
    data.save(file=path_metadata, data=metadata, metadata=metametadata,
              extract=True)

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
    # Example:
    # input_0 is effectively called here without arguments
    V = 1.
    # input_1 is effectively called here with one argument
    for j in range(6):
        # input_2 is effectively called here with two arguments
        for i in range(2):
            # input_3 is effectively called here with three arguments
            for samples in range(100, 1000, 100):
                # input_4 is effectively called here with four arguments
                for num_sim in range(5):
                    # input_5 is effectively called here with five arguments
                    yield (V, j, i, samples, num_sim)


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
def input_0():
    # (Poor) example (you probably don't want to return None)
    return None,  # Note the ',' (comma sign) which makes the output a tuple


@lru_cache(maxsize=1)
def input_3(V, j, i):
    # Example of setting up the hamiltonian.
    return [lipkin_quasi_spin.hamiltonian(V, j)[i]]


# TODO: add your defined input functions to the dictionary below. The key is
#  how many elements from the identifier the input function will be called
#  with.
input_functions = {0: input_0,
                   3: input_3}


# TODO: the function that runs e.g. smallest or vqe.run. Make sure to create
#  non-multiprocess-safe objects (such as VQE-objects) here rather than
#  in the input_n functions. Let id be an identifier; the arguments to this
#  function will then be:
#  id[0], id[1], ..., input_0()[0], input_0()[1], ..., input_1(id[0])[0],
#  input_1(id[0])[1], ..., ..., input_N(id[0], id[1], ..., id[N-1])[0],
#  input_N(id[0], id[1], ..., id[N-1])[1], ...
#  (only including the defined input functions)
def simulate(V, j, i, samples, dummy, h):
    # Use a broad try-except to don't crash if we don't have to
    try:
        # TODO: create VQE-object here! (not multiprocess safe)
        # TODO: run e.g. smallest here and return result.
        result = {'x': j+i+samples, 'fun': j*i*samples}
        return result

    except Exception as e:
        # This will be saved in the metadata file so we can check success-rate.
        return e


# TODO: function that takes in identifier and outputs the file (as a string)
#  to save to. Keep the number of files down!
def file_from_id(identifier):
    return join(directory, file + f'_V{identifier[0]}')


# TODO: function that takes in identifier and outputs metadata-string.
def metadata_from_id(identifier):
    return {'description': f'Simulations with V={identifier[0]}'}


class Bookkeeper:
    """
    Class for keeping track of what's been done and only assign new tasks.
    """
    def __init__(self, iterator, book, output_calc=None):
        """

        :param iterator: Iterable
        :param set book: Set of identifiers corresponding to previously
            completed tasks.
        :param output_calc: List of functions
        """
        if output_calc is None:
            output_calc = {}
        self.iterator = iterator
        self.book = book
        self.output_calc = output_calc

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            x = self.iterator.__next__()
            if x not in self.book:
                output = []
                for i in range(len(x)+1):
                    if i in self.output_calc:
                        output.extend(self.output_calc[i](
                            *[y for j, y in enumerate(x) if j < i]))
                return x, output


def wrap(x):
    return x[0], simulate(*x[0], *x[1])


# Might want to change these to improve performance
num_workers = os.cpu_count()
max_task = 1
chunksize = 1


generator = Bookkeeper(identifier_generator(), ids, input_functions)
files = set()

with Pool(num_workers, maxtasksperchild=max_task) as p:
    result_generator = p.imap_unordered(wrap, generator, chunksize=chunksize)
    for identifier, result in result_generator:
        # Handle exceptions:
        if isinstance(result, Exception):
            # Save the error
            data.append(path_metadata, [identifier, result])
        else:
            file = file_from_id(identifier)
            if file not in files:
                if isfile(file):
                    files.add(file)
                else:
                    # Create file
                    metadata = metadata_from_id(identifier)
                    data.save(file, [], metadata, extract=True)

            # TODO: Save results and identifier
            #  Note that the results will be unordered so make sure to save
            #  enough info!
            data.append(file, list(identifier) + result)

            # Mark the task as completed (last in the else, after saving result)
            data.append(path_metadata, [identifier, True])


# Post simulation.
print('Simulation completed.')
metadata = data.load(path_metadata)[0]
print(f'Total number of tasks {len(metadata)}')
print(f'Number of previously completed tasks: {len(ids)}')
done = sum(x[1] for x in metadata if x[1] is True)
print(f'Number of completed tasks this run: {done - len(ids)}')
print(f'Number of tasks remaining: {len(metadata) - done}')
