from core.lipkin_quasi_spin import hamiltonian, eigs
# from grove.pyvqe.vqe import VQE
# from grove.pyvqe.vqe import expectation_from_sampling
from core import vqe_override
import numpy as np
from pyquil import get_qc
from scipy.optimize import minimize
from core import ansatz
from core import matrix_to_op
from core import vqe_eig
from core import init_params
import matplotlib.pyplot as plt

samples = 10000
j = 2
V = 1
h = hamiltonian(j, V)[0]
print(h.toarray())
eigvals = eigs(j, V)[0]
print(eigvals)
qc = get_qc('3q-qvm')
H = matrix_to_op.multi_particle(h)
print(H)
vqe = vqe_override.VQE_override(minimizer=minimize,
                                minimizer_kwargs={'method': 'Nelder-Mead'})

state = ansatz.multi_particle(init_params.alternate(h.shape[0]))

total_exp = []
total_var = []
for i in range(10):
    min_eig_exp, vars_ = vqe_override.expectation_from_sampling(state, [0],
                                                                samples=10000,
                                                                qc=qc)
    total_exp.append(min_eig_exp)
    total_var.append(vars_)

print('Expected variance:', np.average(total_var))
print('Calculated variance: ', np.var(total_exp))
