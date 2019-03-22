from lipkin_quasi_spin import hamiltonian, eigenvalues
from grove.pyvqe.vqe import VQE
from vqe_eig import smallest_eig_vqe
import time
from pyquil import get_qc
from ansatz import one_particle_ansatz
import numpy as np
from scipy.optimize import minimize
from matrix_to_operator import matrix_to_operator_1

j = 2
V = 1
h = hamiltonian(j, V)
h = h[1]
print(h.shape[0])
qc = get_qc("2q-qvm")
start = time.time()

initial_params = 1/np.sqrt(h.shape[0])*np.array([1 for i in range(h.shape[0]-1)])
disp_opt=False
compare_eig=False
xatol=1e-2
fatol=1e-3
maxiter=1000
Option = {'disp': disp_opt, 'xatol': xatol, 'fatol': fatol, 'maxiter': maxiter, 'options': Option}
vqe = VQE(minimizer=minimize, minimizer_kwargs={'method': 'Nelder-Mead'})
H = matrix_to_operator_1(h)
eig = vqe.vqe_run(one_particle_ansatz, H, initial_params, samples=1, qc=qc)
#eig = smallest_eig_vqe(h, one_particle_ansatz, qc, num_samples=None, opt_algorithm='Nelder-Mead')[0]
print(eig)
end = time.time()
print('Time Taken: ', end - start)
