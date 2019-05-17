"""
Recipe for long runs. This makes sure to save often and keeps the data in a
single file.

@author = Joel, Carl
"""
import core.interface
from os.path import basename
from functools import lru_cache
from core.interface import hamiltonians_of_size, vqe_nelder_mead, vqe_bayes
from core import vqe_eig, parallel
from core import callback as cb
import numpy as np
import sys


#4x4 max_meas=3e6, samples=68e3
#3x3 max_meas=3e6, samples=132e3
#2x2 max_meas=


# Input number of workers
run_kwargs = parallel.script_input(sys.argv)

version = 3

directory = 'hero_run_nm'  # directory to save to


def identifier_generator():
    max_meas = 0.5e6
    for size, samples in zip([2], [64e3]):
        for V in np.linspace(0, 1, 10):
            for hamiltonian_idx in [1, 2]:
                yield (size, hamiltonian_idx, V, int(max_meas), int(samples))


@lru_cache(maxsize=1)
def input_3(size, hamiltonian_idx, V):
    if V == np.inf:
        e = 0.
        V = 1.
    else:
        e = 1
    h, eig = hamiltonians_of_size(size, V, e)
    return h[hamiltonian_idx], eig[hamiltonian_idx]


@lru_cache(maxsize=1)
def input_5(size, hamiltonian_idx, V, max_meas, samples):
    print(f'size={size}, Hamiltonian_idx={hamiltonian_idx}, V={V}'
          f'max_meas={max_meas}, samples={samples}')
    return ()


input_functions = {3: input_3,
                   5: input_5}


def simulate(size, hamiltonian_idx, V, max_meas, samples, h, eig):

    H, qc, ansatz_, initial_params = \
        core.interface.create_and_convert('multi_particle', h)

    vqe = vqe_nelder_mead(samples=samples, H=H, fatol=0, xatol=0)
    tol_para = 1e-3
    max_same_para = 3
    callback = cb.restart(max_same_para, tol_para)

    result = vqe_eig.smallest(H, qc, initial_params, vqe,
                              ansatz_, samples,
                              callback=callback, max_meas=max_meas)
    result.correct = eig
    return result


def file_from_id(identifier):
    return f'size={identifier[0]}'


def metadata_from_id(identifier):
    return {'description': 'Data for hero run.',
            'identifier_description': ['size', 'hamiltonian_idx', 'V',
                                       'max_meas', 'samples'],
            'max_same_para_nm': 3,
            'tol_para_nm': 1e-3,
            'size': identifier[0]}


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
