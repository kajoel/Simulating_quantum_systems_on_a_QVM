# Carl 4/4
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
from meas import sweep
import matplotlib.pyplot as plt


###############################################################################
# Functions
def save_data(file, data_, metadata):
    data.save(file, data_, metadata)


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

    return H, qc, ansatz_, initial_params, ansatz_name


def fatol_var(ansatz_, initial_params, H, samples, qc):
    vqe = vqeOverride.VQE_override(minimizer=minimize,
                                   minimizer_kwargs={'method': 'NelderMead'})

    _, var = vqe.expectation(ansatz_(initial_params), H, samples=samples, qc=qc)
    return 2 * np.sqrt(var)


###############################################################################
# Parameters
j, V, matrix = 1, 1, 0
h = hamiltonian(j, V)[matrix]
dim = h.shape[0]
eig = eigs(j, V)[matrix][0]
print(h.toarray(), '\nEigenvalue:', eig, '\n')
samples = 2000
###############################################################################
# Ansatz
ansatz_types = ['one_particle', 'one_particle_ucc', 'multi_particle',
                'multi_particle_ucc']

H, qc, ansatz_, initial_params, ansatz_name = ansatz_type(ansatz_types[1], h, dim)
###############################################################################
# Facit paramater
facit = vqe_eig.smallest(H, qc, initial_params, ansatz_, disp=False)[1]
###############################################################################
# Tol
fatol = fatol_var(ansatz_, initial_params, H, samples, qc)
xatol = 1e-2
tol_para = 1e-2
max_para = 4
max_iter = 20
increase_samples = 0
parameters = {'fatol': fatol, 'xatol': xatol, 'tol_para': tol_para,
              'max_para': max_para, 'max_iter': max_iter,
              'increase_samples': increase_samples}
###############################################################################
# Result
result = vqe_eig.smallest_restart(H, qc, initial_params, ansatz_, samples,
                                  max_para=max_para,
                                  max_iter=max_iter,
                                  tol_para=tol_para,
                                  increase_samples=increase_samples,
                                  xatol=xatol,
                                  fatol=fatol)
###############################################################################
print('Facit parameter:', facit, '\n')
print(result)

file = 'NelderMead_Restart_1.pkl'
data_ = result
data_['facit'] = facit
metadata = {'info': 'Test av NelderMead Restart', 'j': j, 'matrix': matrix,
            'ansatz': ansatz_name, 'H': H, 'len_H': len(H),
            'paramaters': parameters}

save_data(file, data_, metadata)
