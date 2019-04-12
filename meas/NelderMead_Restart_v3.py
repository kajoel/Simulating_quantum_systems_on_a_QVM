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


###############################################################################
# Parameters
V, matrix = 1, 0
ansatz_types = ['one_particle', 'one_particle_ucc', 'multi_particle',
                'multi_particle_ucc']
xatol = 1e-2
tol_para = 1e-2
max_iter = 100
increase_samples = 0

max_params = range(3, 6)
samples = np.linspace(500, 10000, 20)
base_dir = join(ROOT_DIR, 'data/NelderMead_Restart_v1')

i = 1

for j, sample, ansatz_name, max_para, k in itertools.product(range(1, 4),
                                                             samples,
                                                             ansatz_types,
                                                             max_params,
                                                             range(1, 4)):

    sample = int(sample)
    h = hamiltonian(j, V)[matrix]
    dim = h.shape[0]
    eig = eigs(j, V)[matrix][0]
    samples = 2000
    H, qc, ansatz_, initial_params = ansatz_type(ansatz_name, h, dim)
    facit = vqe_eig.smallest(H, qc, initial_params, ansatz_, disp=False)[1]
    fatol = fatol_var(ansatz_, initial_params, H, sample, qc)

    print('\nLoop: j = {}, ansatz_name = {}, samples = {}, '
          'max_para = {}, k = {}, i = {}'.format(j, ansatz_name, sample,
                                                 max_para, k, i))

    result = vqe_eig.smallest_restart(H, qc, initial_params, ansatz_, sample,
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

    file = 'NelderMead_Restart_j={}_samples={}_i={}_k={}.pkl'.format(j, sample,
                                                                     i, k)
    data_ = result
    data_['facit'] = facit
    data_['samples'] = sample

    metadata = {'info': 'NelderMead Restart', 'j': j, 'matrix': matrix,
                'ansatz': ansatz_name, 'initial_params': initial_params, 'H': H,
                'len_H': len(H),
                'paramaters': parameters}

    save_data(file, data_, metadata, base_dir)
    if k == 3: i += 1
