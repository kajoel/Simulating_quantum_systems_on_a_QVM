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
samples = 1000
matrix = 0
j = 3
V = 1
h = hamiltonian(j, V)[matrix]
dim = h.shape[0]
print(h.toarray())
print('\n')
eigvals = eigs(j, V)[matrix]
print(eigvals)
print('\n')
#qc = get_qc(str(int.bit_length(h.shape[0])) + 'q-qvm')
qc = get_qc(str(h.shape[0]) + 'q-qvm')
###############################################################################
H = matrix_to_op.one_particle(h)
initial_params = init_params.ones(dim)
ansatz_ = ansatz.one_particle_ucc(dim)

eig = vqe_eig.smallest_restart(H, qc, initial_params, ansatz_, samples, xatol=1e-2, fatol=1e-3)