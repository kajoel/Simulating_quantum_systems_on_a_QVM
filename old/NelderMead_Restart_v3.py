# Carl 10/4
###############################################################################
# Imports
import core.interface
from core.lipkin_quasi_spin import hamiltonian, eigs
from grove.pyvqe.vqe import VQE
import numpy as np
from pyquil import get_qc
from scipy.optimize import minimize
from core import ansatz
from core import matrix_to_op
from core import vqe_eig
from core import init_params
from core import vqe_override
from core import data
import itertools
from constants import ROOT_DIR
from os.path import join
from core import callback as cb
from core import create_vqe


###############################################################################
# Functions
def save_data(file, data_, metadata, base_dir):
    data.save(file, data_, metadata, base_dir)


def fatol_var(ansatz_, initial_params, H, samples, qc):
    vqe = vqe_override.VQE_override(minimizer=minimize,
                                    minimizer_kwargs={'method': 'Nelder-Mead'})

    _, var = vqe.expectation(ansatz_(initial_params), H, samples=samples, qc=qc)
    return 2 * np.sqrt(var)


def fatol_skattning(sample, H):
    return sum(np.abs(term.coefficient) for term in H.terms) / np.sqrt(sample)


###############################################################################
# Parameters
V, matrix = 1, 0
ansatz_types = ['one_particle', 'one_particle_ucc', 'multi_particle',
                'multi_particle_ucc']

ansatz_types = ['one_particle_ucc', 'multi_particle']

xatol = 1e-2
tol_para = 1e-2
max_iter = 20
increase_samples = 0

max_params = range(5, 10)
samples = np.linspace(500, 10000, 20)
base_dir = join(ROOT_DIR, 'data/NelderMead_Restart_v2')

iters = 5

for j, ansatz_name in itertools.product(range(1, 6), ansatz_types):

    h = hamiltonian(j, V)[matrix]
    dim = h.shape[0]
    eig = eigs(j, V)[matrix][0]
    H, qc, ansatz_, initial_params = core.interface.create_and_convert(ansatz_name, h, dim)
    print(ansatz_)
    print(initial_params)
    samples = np.linspace(500, 10000 * len(H), 100)

    for sample, max_para in itertools.product(samples, max_params):

        sample = int(round(sample))

        fatol = fatol_skattning(sample, H)

        data_ = {}

        disp_options = {'disp': False, 'xatol': xatol, 'fatol': fatol,
                        'return_all': True, 'maxiter': 10000}

        vqe = vqe_override.VQE_override(minimizer=minimize,
                                        minimizer_kwargs={'method':
                                                              'nelder-mead',
                                                          'options': disp_options})

        for iter in range(iters):
            print('\nLoop: j = {}, ansatz_name = {}, samples = {}, \
max_para = {}, fatol = {}, iteration = {}/{}' \
                  .format(j, ansatz_name, sample, max_para, round(fatol, 3),
                          iter + 1, iters))

            #vqe = create_vqe.default_nelder_mead()
            callback = cb.restart(2, tol_para, True)

            facit = vqe_eig.smallest(H, qc, initial_params, vqe, ansatz_)['fun']

            result = vqe_eig.smallest(H, qc, initial_params, vqe, ansatz_,
                                       sample, disp=True, callback=callback, attempts = max_iter)
            print(result)
            parameters = {'fatol': fatol, 'xatol': xatol, 'tol_para': tol_para,
                          'max_para': max_para, 'max_iter': max_iter,
                          'increase_samples': increase_samples}

            result['facit'] = facit
            result['samples'] = sample
            data_[iter] = result

        file = 'NelderMead_Restart_j={}_samples={}_ansatz={}_maxpara={}.pkl' \
            .format(j, sample, ansatz_name, max_para)

        metadata = {'info': 'NelderMead Restart', 'j': j, 'matrix': matrix,
                    'ansatz': ansatz_name, 'initial_params': initial_params,
                    'H': H, 'len_H': len(H), 'paramaters': parameters, }

        save_data(file, data_, metadata, base_dir)
