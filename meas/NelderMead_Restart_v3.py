# Carl 10/4
###############################################################################
# Imports
from core.lipkin_quasi_spin import hamiltonian, eigs
from grove.pyvqe.vqe import VQE
import numpy as np
from pyquil import get_qc
from scipy.optimize import minimize
from core import ansatz
from core import matrix_to_op
from core import vqe_eig
from core import init_params
from core import vqeOverride
from core import data
import itertools
from constants import ROOT_DIR
from os.path import join


###############################################################################
# Functions
def save_data(file, data_, metadata, base_dir):
    data.save(file, data_, metadata, base_dir)


def ansatz_type(ansatz_name, h, dim):
    if ansatz_name == 'one_particle':
        qc = get_qc(str(h.shape[0]) + 'q-qvm')
        H = matrix_to_op.one_particle(h)
        ansatz_ = ansatz.one_particle(dim)
        initial_params = init_params.alternate(dim)

    elif ansatz_name == 'one_particle_ucc':
        qc = get_qc(str(h.shape[0]) + 'q-qvm')
        H = matrix_to_op.one_particle(h)
        ansatz_ = ansatz.one_particle_ucc(dim)
        initial_params = init_params.ucc(dim)

    elif ansatz_name == 'multi_particle':
        qc = get_qc(str(int.bit_length(h.shape[0])) + 'q-qvm')
        H = matrix_to_op.multi_particle(h)
        ansatz_ = ansatz.multi_particle(dim)
        initial_params = init_params.alternate(dim)

    elif ansatz_name == 'multi_particle_ucc':
        qc = get_qc(str(int.bit_length(h.shape[0])) + 'q-qvm')
        H = matrix_to_op.multi_particle(h)
        ansatz_ = ansatz.multi_particle_ucc(dim)
        initial_params = init_params.ucc(dim)

    return H, qc, ansatz_, initial_params


def fatol_var(ansatz_, initial_params, H, samples, qc):
    vqe = vqeOverride.VQE_override(minimizer=minimize,
                                   minimizer_kwargs={'method': 'NelderMead'})

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

i = 1

for j, ansatz_name in itertools.product(range(1, 6), ansatz_types):

    h = hamiltonian(j, V)[matrix]
    dim = h.shape[0]
    eig = eigs(j, V)[matrix][0]
    H, qc, ansatz_, initial_params = ansatz_type(ansatz_name, h, dim)

    samples = np.linspace(500, 10000 * len(H), 100)

    for sample, max_para, iter in itertools.product(samples, max_params,
                                                    range(1, iters)):

        sample = int(round(sample))

        facit = vqe_eig.smallest(H, qc, initial_params, ansatz_, disp=False)[1]
        fatol = fatol_skattning(sample, H)

        data_ = {}

        for iter in range(iters):
            print('\nLoop: j = {}, ansatz_name = {}, samples = {}, \
max_para = {}, fatol = {}, iteration = {}/{}'\
            .format( j, ansatz_name, sample, max_para, round(fatol,3), iter + 1, iters))

            result = vqe_eig.smallest_restart(H, qc, initial_params, ansatz_,
                                              sample,
                                              max_para=max_para,
                                              max_iter=max_iter,
                                              tol_para=tol_para,
                                              increase_samples=increase_samples,
                                              xatol=xatol,
                                              fatol=fatol,
                                              disp=False,
                                              disp_iter=False)

            parameters = {'fatol': fatol, 'xatol': xatol, 'tol_para': tol_para,
                          'max_para': max_para, 'max_iter': max_iter,
                          'increase_samples': increase_samples}

            result['facit'] = facit
            result['samples'] = sample
            data_[iter] = result

        file = 'NelderMead_Restart_j={}_samples={}_maxpara={}.pkl' \
            .format(j, sample, max_para)

        metadata = {'info': 'NelderMead Restart', 'j': j, 'matrix': matrix,
                    'ansatz': ansatz_name, 'initial_params': initial_params,
                    'H': H, 'len_H': len(H), 'paramaters': parameters, }

        save_data(file, data_, metadata, base_dir)
