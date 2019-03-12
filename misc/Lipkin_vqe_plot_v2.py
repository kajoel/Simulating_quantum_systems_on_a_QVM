#Carl, 2019-03-08
#########################################
import sys
sys.path.insert(0, './')
#########################################
import numpy as np
import matplotlib.pyplot as plt
#########################################
from grove.pyvqe.vqe import VQE
import pyquil.api as api
from scipy.optimize import minimize
##########################################
import lipkin_quasi_spin as lqs
from matrix_to_pyquil import matrix_to_pyquil
import ansatz as ansatz
#################################################################################
vqe = VQE(minimizer=minimize,minimizer_kwargs={'method': 'L-BFGS-B'})
#vqe = VQE(minimizer=minimize,minimizer_kwargs={'method': 'Nelder-Mead'})
qvm = api.QVMConnection()
#################################################################################
def vqe_eig(vqe, h, qvm):
    H = matrix_to_pyquil(h)
    initial_params = 1 / np.sqrt(h.shape[0]) * np.ones([h.shape[0]])
    result = vqe.vqe_run(ansatz.one_particle_ansatz, H, initial_params, samples=None, qvm=qvm)
    return result
#################################################################################
a = 10
V = np.linspace(0,1,a)

eig1 = np.zeros([a])
eig2 = np.zeros([a])
eig3 = np.zeros([a])
eig4 = np.zeros([a])

kvant_j = 4

for k,v in enumerate(V):
    (h1, h2) = lqs.hamiltonian(kvant_j, v)

    result = vqe_eig(vqe,h1,qvm)
    eig1[k] = np.abs(result['fun'])

    v = result['x']
    h1 += 1.1 * np.abs(result['fun']) * np.outer(v,v)

    result = vqe_eig(vqe,h1,qvm)
    eig2[k] = np.abs(result['fun'])

    result = vqe_eig(vqe,h2,qvm)
    eig3[k] = np.abs(result['fun'])

    v = result['x']
    h2 += 1.1 * np.abs(result['fun']) * np.outer(v,v)

    result = vqe_eig(vqe,h2,qvm)
    eig4[k] = np.abs(result['fun'])

plt.plot(V,eig1)
plt.plot(V,eig2)
plt.plot(V,eig3)
plt.plot(V,eig4)

plt.show()

#tikz('Lipkin_vqe_j4_1.tex')