# Sebastian, Carl 4/4
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
from core import vqeOverride

###############################################################################
samples = 1000
matrix = 0
j = 4
V = 1
h = hamiltonian(j, V)[matrix]
dim = h.shape[0]
print(h.toarray())
print('\n')
eigvals = eigs(j, V)[matrix]
print(eigvals)
print('\n')
qc = get_qc(str(int.bit_length(h.shape[0])) + 'q-qvm')
#qc = get_qc(str(h.shape[0]) + 'q-qvm')
###############################################################################
H = matrix_to_op.multi_particle(h)
initial_params = init_params.alternate(dim)
ansatz_ = ansatz.multi_particle(dim)

vqe = vqeOverride.VQE_override(minimizer=minimize,
                               minimizer_kwargs={'method': 'Nelder-Mead'})

_, var = vqe.expectation(ansatz_(initial_params), H, samples=samples, qc=qc)
fatol = 2 * np.sqrt(var)
print(fatol)
eig = vqe_eig.smallest_restart(H, qc, initial_params, ansatz_, samples,
                               xatol=1e-2, fatol=fatol, return_all_data=True)
print(eig)


eig = vqe.expectation(ansatz_(eig['x']), H, samples=samples, qc=qc)
print(eig)