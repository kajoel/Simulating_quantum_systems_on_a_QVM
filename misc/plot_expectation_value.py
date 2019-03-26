#Carl 12/3
###############################################################################
#Imports
from pyquil import Program, get_qc
import time
import numpy as np
from scipy.optimize import minimize
from grove.pyvqe.vqe import VQE
from matplotlib import pyplot as plt
from datetime import datetime,date
from matplotlib import cm

# Imports from our projects
from matrix_to_operator import matrix_to_operator_1
from lipkin_quasi_spin import hamiltonian,eigs
from ansatz import one_particle as ansatz
from ansatz import one_particle_inital, carls_initial
from vqe_eig import smallest as vqe_eig

###############################################################################
samples = 1000
matrix = 0
j = 2
V = 1
h = hamiltonian(j, V)
h = h[0]
qc = get_qc(str(h.shape[0]) + 'q-qvm')
###############################################################################
eig_CPU = eigs(j, V)
print(eig_CPU[0])
#eig_CPU = eig_CPU[1]
#start = time.time()
#end = time.time()
#print('Tid VQE: ' + str(end-start))

vqe = VQE(minimizer=minimize, minimizer_kwargs={'method': 'Nelder-Mead'})
H = matrix_to_operator_1(h)
eig = 0
#for i in range(50):
    #eig += vqe.expectation(ansatz(result['x']), H, samples=20000, qc=qc)

#eig = eig/50
#print(eig)
#plt.plot(result['x'],eig,'*', ms=10)
#plt.show()
###############################################################################
result = vqe_eig(h, qc_qvm=qc, initial=carls_initial(h.shape[0]),
                 num_samples=None, disp_run_info=True, display_after_run=True,
                 xatol=1e-3, fatol=5e-1, return_all_data=True)
for i in range(20):
    eig += vqe.expectation(ansatz(result['x']), H, samples=500, qc=qc)

eig = eig/20
print(result['fun'])
print(eig)