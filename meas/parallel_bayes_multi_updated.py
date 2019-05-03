"""
Recipe for long runs. This makes sure to save often and keeps the data in a
single file.

@author = Joel, Carl, Axel
"""
import core.interface
from os.path import basename
from functools import lru_cache
from core.interface import hamiltonians_of_size, vqe_bayes
from core import vqe_eig, parallel
import numpy as np
import sys

# TODO: When writing a meas script, change (only) the parts marked by TODOs.
#  MAKE SURE TO SAFE ENOUGH INFORMATION!
#  BETTER TO SAVE TOO MUCH THAN TOO LITTLE!

# Input number of workers
run_kwargs = parallel.script_input(sys.argv)

# TODO: give a version-number of the script (this should be changed iff the
#  meaning of the elements in the tuple yielded by the generator (
#  identifier_generator) changes. Changing this will start a completely new
#  simulation (the default behaviour of this script is to continue where it
#  stopped last time it was run). The version-number will be added to the
#  file name (e.g test_v1_...)
version = 3

# TODO: select directory to save to.
directory = 'bayes_total_evals'  # directory to save to


# TODO: identifiers to iterate over, should yield a tuple that uniquely
#  identifies the simulation (using floats/ints as elements is a good idea,
#  e.g (1, 2, 3, 4)), i.e. every call to smallest, vqe.run or similar should
#  have a unique identifier.
def identifier_generator():
    # ansatz
    ansatz_name = 'multi_particle'
    # size of hamiltonian
    for size in range(2, 6):
        num_para = size-1
        for hamiltonian_idx in range(4):
            # number of calls
            for n_calls in range(10, 30 + num_para*15, 5 + 5*(num_para-1)):
                # number of samples
                for samples in range(100, 250 + 4000*num_para, 250*num_para):
                    # input_4 is effectively called here with four arguments
                    for repeats in range(5):
                        # input_5 is effectively called here with five arguments
                        yield (ansatz_name, size, hamiltonian_idx,
                               int(round(samples)), n_calls, repeats)


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
def input_5(ansatz_name, size, hamiltonian_idx, samples, n_calls):
    print(f'Size={size}, Hamiltonian_idx={hamiltonian_idx}, '
          f'Samples={samples}, n_calls={n_calls}')
    return ()


# TODO: add your defined input functions to the dictionary below. The key is
#  how many elements from the identifier the input function will be called
#  with.
input_functions = {3: input_3,
                   5: input_5}


# TODO: the function that runs e.g. smallest or vqe.run. Make sure to create
#  non-multiprocess-safe objects (such as VQE-objects) here rather than
#  in the input_n functions. Let id be an identifier; the arguments to this
#  function will then be:
#  id[0], id[1], ..., input_0()[0], input_0()[1], ..., input_1(id[0])[0],
#  input_1(id[0])[1], ..., ..., input_N(id[0], id[1], ..., id[N-1])[0],
#  input_N(id[0], id[1], ..., id[N-1])[1], ...
#  (only including the defined input functions)
def simulate(ansatz_name, size, hamiltonian_idx, samples, n_calls,
             repeats, h, eig):
    # TODO: create VQE-object here! (not multiprocess safe)
    # TODO: run e.g. smallest here and return result.
    interval = [(-1.0, 1.0)]*(size-1)
    H, qc, ansatz_,_ = core.interface.create_and_convert(ansatz_name, h)
    vqe = vqe_bayes(n_calls=n_calls)
    result = vqe_eig.smallest(H, qc, interval, vqe, ansatz_, samples)

    result.correct = eig
    return result


# TODO: function that takes in identifier and outputs the file (as a string)
#  to save to. Keep the number of files down!
def file_from_id(identifier):
    return f'{identifier[0]}_size={identifier[1]}_matidx={identifier[2]}'


# TODO: function that takes in identifier and outputs metadata-string.
def metadata_from_id(identifier):
    return {'description': 'Bayes optimizer, identifier = data[:][0], '
                           'result = data[:][1]',
            'size': identifier[1],
            'matidx': identifier[2],
            'ansatz': identifier[0]}


parallel.run(
    simulate=simulate,
    identifier_generator=identifier_generator,
    input_functions=input_functions,
    directory=directory,
    version=version,
    script_file=basename(__file__),
    file_from_id=file_from_id,
    metadata_from_id=metadata_from_id,
    **run_kwargs
)
