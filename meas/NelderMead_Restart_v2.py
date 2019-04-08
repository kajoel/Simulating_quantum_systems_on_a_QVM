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
from meas import sweep
import matplotlib.pyplot as plt

###############################################################################
j, V, matrix = 1, 1, 0
h = hamiltonian(j, V)[matrix]
dim = h.shape[0]
eig = eigs(j, V)[matrix][0]

print(h.toarray(), '\n', 'Eigenvalue:', eig, '\n')

eig_tol = 2e-2
para_tol = 4e-2
parameters = []
variances = []


def ansatz_type(ansatz_):
    if ansatz_ == 'one_particle':
        qc = get_qc(str(h.shape[0]) + 'q-qvm')
        H = matrix_to_op.one_particle(h)
        ansatz_ = ansatz.one_particle(dim)
        initial_params = init_params.alternate(dim)

    elif ansatz_ == 'one_particle_ucc':
        qc = get_qc(str(h.shape[0]) + 'q-qvm')
        H = matrix_to_op.one_particle(h)
        ansatz_ = ansatz.one_particle_ucc(dim)
        initial_params = init_params.ucc(dim)

    elif ansatz_ == 'multi_particle':
        qc = get_qc(str(int.bit_length(h.shape[0])) + 'q-qvm')
        H = matrix_to_op.multi_particle(h)
        ansatz_ = ansatz.multi_particle(dim)
        initial_params = init_params.alternate(dim)

    elif ansatz_ == 'multi_particle_ucc':
        qc = get_qc(str(int.bit_length(h.shape[0])) + 'q-qvm')
        H = matrix_to_op.multi_particle(h)
        ansatz_ = ansatz.multi_particle_ucc(dim)
        initial_params = init_params.ucc(dim)

    return H, qc, ansatz_, initial_params


a = 10
samples = np.linspace(1000, 10000, a)
samples = 1000
len_exp_vals = []

xatol, fatol = 1e-2, 1e-2

ansatz_types = ['one_particle', 'one_particle_ucc', 'multi_particle',
                'multi_particle_ucc']

H, qc, ansatz_, initial_params = ansatz_type(ansatz_types[1])

result = vqe_eig.smallest_restart(H, qc, initial_params, ansatz_, samples,
                                  max_para=2, tol_para=1e-2, increase_samples=0,
                                  xatol=xatol, fatol=fatol,
                                  return_all_data=True)

print(result)
