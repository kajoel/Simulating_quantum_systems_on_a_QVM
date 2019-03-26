#Carl 12/3
###############################################################################
#Imports
from core.lipkin_quasi_spin import hamiltonian, eigs
from grove.pyvqe.vqe import VQE
import numpy as np
from pyquil import get_qc
from scipy.optimize import minimize
from core import ansatz
from core import matrix_to_op
from core import vqe_eig
from core import init_params
import matplotlib.pyplot as plt
###############################################################################
samples = 10000
matrix = 0
j = 2
V = 1
h = hamiltonian(j, V)[matrix]
print(h.toarray())
print('\n')
eigvals = eigs(j, V)[matrix]
print(eigvals)
print('\n')
qc = get_qc(str(int.bit_length(h.shape[0])) + 'q-qvm')
###############################################################################
H = matrix_to_op.multi_particle(h)

vqe_eig.smallest(h, qc=qc, ansatz_=ansatz.multi_particle, samples=samples,
                 fatol=1e-2, initial=init_params.alternate(h.shape[0]),
                 disp_run_info=True)
###############################################################################
