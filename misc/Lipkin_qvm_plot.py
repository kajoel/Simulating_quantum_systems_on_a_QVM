from openfermion.transforms import jordan_wigner,get_sparse_operator, get_fermion_operator
from openfermion.ops import FermionOperator, QubitOperator
from forestopenfermion import pyquilpauli_to_qubitop, qubitop_to_pyquilpauli
import numpy as np
from scipy.linalg import eigh
from pyquil.paulis import sZ, exponentiate
from pyquil.quil import Program
from pyquil import get_qc
import pyquil.api as api
from pyquil.gates import *
from grove.pyvqe.vqe import VQE
from grove.alpha.arbitrary_state.arbitrary_state import create_arbitrary_state
from grove.alpha.arbitrary_state.unitary_operator import fix_norm_and_length
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from quasi_spin_hamiltonian import quasi_spin_hamiltonian
from matrix_to_pyquil import matrix_to_pyquil
import scipy.sparse
from scipy.sparse.linalg import eigsh
import itertools
from matplotlib2tikz import save as tikz
###################################################################################
vqe = VQE(minimizer=minimize,minimizer_kwargs={'method': 'L-BFGS-B'})
#vqe = VQE(minimizer=minimize,minimizer_kwargs={'method': 'Nelder-Mead'})
qvm = api.QVMConnection()
###################################################################################
#UCC ANSATZ
def UCC_ansatz(theta):
    #theta = fix_norm_and_length(theta)
    program = Program()
    #ro = program.declare('ro', memory_type='BIT', memory_size=1)
    vector = np.zeros(2**(theta.shape[0]))
    vector[[2**i for i in range(theta.shape[0])]] = theta
    program += create_arbitrary_state(vector)
    #program += MEASURE(0, ro[0])
    return program
###################################################################################
#GROVE VQE_RUN
def vqe_eig(vqe, h, qvm):
    H = matrix_to_pyquil(h)
    initial_params = 1 / np.sqrt(h.shape[0]) * np.ones([h.shape[0]])
    result = vqe.vqe_run(UCC_ansatz, H, initial_params, samples=None, qvm=qvm)
    return result
###################################################################################
a = 10
V = np.linspace(0,1,a)

eig1 = np.zeros([a])
eig2 = np.zeros([a])
eig3 = np.zeros([a])
eig4 = np.zeros([a])

kvant_j = 5

for k,v in enumerate(V):
    (h1, h2) = quasi_spin_hamiltonian(kvant_j, v)

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