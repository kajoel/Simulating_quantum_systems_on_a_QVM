# Carl 12/3
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
samples = 500
matrix = 0
j = 1
V = 1
h = hamiltonian(j, V)[matrix]
dim = h.shape[0]
print(h.toarray())
print('\n')
eigvals = eigs(j, V)[matrix]
print(eigvals)
print('\n')
# qc = get_qc(str(int.bit_length(h.shape[0])) + 'q-qvm')
qc = get_qc(str(h.shape[0]) + 'q-qvm')
###############################################################################
H = matrix_to_op.one_particle(h)
initial_params = init_params.alternate(dim)
ansatz_ = ansatz.one_particle(dim)
# ansatz_ = ansatz.one_particle_ucc(dim, trotter_order=1, trotter_steps=4)
###############################################################################
'''
parameter_None = vqe_eig.smallest(H, qc, initial_params, ansatz_=ansatz_,
                 samples=None, fatol=1e-1, xatol=7e-2, disp_run_info = True,
                                  display_after_run=True)[1]
'''

# print('\n', 'Paramater (samples = None):', parameter_None, '\n')
###############################################################################
# eigs, parameters = sweep.sweep(h, qc, ansatz_, matrix_to_op.one_particle, start=-100, stop=30)
# plt.plot(parameters, eigs)
# plt.show()
dimension = [(-2, 2)] * (dim - 1)
num = 40
###############################################################################
run_data = vqe_eig.smallest_bayes(H, qc, dimension, ansatz_,
                                       samples, return_all_data=True,
                                       n_calls=num)


###############################################################################
print(run_data)

'''
parameter = result[1]
ansatz_ = ansatz_(parameter)
vqe = VQE(minimizer=minimize, minimizer_kwargs={'method': 'Nelder-Mead'})



n=20
eig = 0
for i in range(n):
    eig += vqe.expectation(ansatz_, H, samples=samples, qc=qc)

eig = eig/n

print('\n', 'Eigenvalue after mean:',eig)
'''
###############################################################################
