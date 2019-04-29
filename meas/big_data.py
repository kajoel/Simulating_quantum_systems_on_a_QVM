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


# Input number of workers
run_kwargs = parallel.script_input(sys.argv)

version = 1

directory = 'final'  # directory to save to


def identifier_generator():
    for repeats in range(5):
        # ansatz
        for ansatz_name in ['multi_particle', 'one_particle_ucc']:
            for minimizer in ['nelder-mead', 'bayes']:
                # size of hamiltonian
                for size in range(5, 6):
                    # the index of the four hamiltonians
                    for hamiltonian_idx in range(4):
                        # number of measurements on qc
                        for max_meas in np.linspace(1e6, 3e6, 51):
                            # number of samples
                            for samples in np.linspace(5e4, 3e5, 51):
                                yield (repeats, ansatz_name, minimizer,
                                       size, hamiltonian_idx,
                                       int(max_meas),
                                       int(samples))


@lru_cache(maxsize=1)
def input_5(repeats, ansatz_name, minimizer, size, hamiltonian_idx):
    h, eig = hamiltonians_of_size(size)
    return h[hamiltonian_idx], eig[hamiltonian_idx]


@lru_cache(maxsize=1)
def input_7(repeats, ansatz_name, minimizer, size, hamiltonian_idx, max_meas,
            samples):
    print(f'Ansatz={ansatz_name}, minimizer={minimizer}, repeat={repeats}, '
          f'size={size}, Hamiltonian_idx={hamiltonian_idx}, '
          f'max_meas={max_meas}, samples={samples}')
    return ()


input_functions = {5: input_5,
                   7: input_7}


def simulate(repeats, ansatz_name, minimizer, size, hamiltonian_idx, max_meas,
             samples, h, eig):

    H, qc, ansatz_, initial_params = \
        core.interface.create_and_convert(ansatz_name, h)

    if minimizer == 'nelder-mead':
        vqe = vqe_nelder_mead(samples=samples, H=H, fatol=0, xatol=0)
        tol_para = 1e-3
        max_same_para = 3
        callback = cb.restart(max_same_para, tol_para)

    elif minimizer == 'bayes':
        n_calls = int(round(max_meas/samples))
        vqe = vqe_bayes(n_calls=n_calls)

        def callback(*args, **kwargs):
            pass

        if ansatz_name == 'multi_particle':
            initial_params = [(-1.0, 1.0)]*(size - 1)
        elif ansatz_name == 'one_particle_ucc':
            initial_params = [(-3.0, 3.0)] * (size - 1)
        else:
            raise RuntimeError("Don't know that ansatz.")
    else:
        raise RuntimeError('Bad minimizer')

    result = vqe_eig.smallest(H, qc, initial_params, vqe,
                              ansatz_, samples,
                              callback=callback, max_meas=max_meas)
    result.correct = eig
    return result


def file_from_id(identifier):
    return f'{identifier[1]}_{identifier[2]}_size={identifier[3]}'


def metadata_from_id(identifier):
    return {'description': 'Data for heatmaps, both Nelder-Mead and Bayes. '
                           'identifier = data[:][0], result = data[:][1]',
            'identifier_description': ['ansatz_name', 'minimizer', 'repeats',
                                       'size', 'hamiltonian_idx', 'max_meas',
                                       'samples'],
            'max_same_para_nm': 3,
            'tol_para_nm': 1e-3,
            'size': identifier[3],
            'matidx': identifier[4],
            'ansatz': identifier[1],
            'minimizer': identifier[2]}


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
