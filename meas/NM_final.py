"""
Recipe for long runs. This makes sure to save often and keeps the data in a
single file.

@author = Joel, Carl
"""
import core.interface
from os.path import basename
from functools import lru_cache
from core.interface import hamiltonians_of_size, vqe_nelder_mead
from core import vqe_eig, parallel
from core import callback as cb
import numpy as np
import sys


# Input number of workers
run_kwargs = parallel.script_input(sys.argv)

version = 1

directory = 'NM_final'  # directory to save to


# TODO: do bayes and NM in same script?l
def identifier_generator():
    # ansatz
    for ansatz_name in ['multi_particle', 'one_particle_ucc']:
        # size of hamiltonian
        for size in range(4, 5):  # TODO: will 5x5 work?!
            # the index of the four hamiltonians
            for hamiltonian_idx in range(4):
                # number of measurements on qc
                for max_meas in np.linspace(0.5e6, 3e6, 100):  # TODO
                    # number of samples
                    for samples in np.linspace(400, 60000, 100):  # TODO
                        for repeats in range(2):  # TODO
                            yield (ansatz_name, size, hamiltonian_idx,
                                   max_meas, int(round(samples)), repeats)


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
    max_meas = 200  # TODO, 2.5e6
    result = vqe_eig.smallest(H, qc, initial_params, vqe,
                              ansatz_, samples,
                              callback=callback, max_meas=max_meas)
    result.correct = eig
    return result


# TODO: function that takes in identifier and outputs the file (as a string)
#  to save to. Keep the number of files down!
def file_from_id(identifier):
    return f'{identifier[0]}_size={identifier[1]}_matidx={identifier[2]}'


# TODO: function that takes in identifier and outputs metadata-string.
def metadata_from_id(identifier):
    return {'description': 'Nelder-Mead Restart, identifier = data[:][0], '
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
