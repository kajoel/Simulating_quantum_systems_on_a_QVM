#Carl 12/3
import numpy as np
import pyquil.gates as qg
from pyquil import Program, get_qc
from scipy.optimize import minimize
from matplotlib import pyplot as plt
from vqe_eig import calculate_eigenvalues_vqe, smallest_eig_vqe
from lipkin_quasi_spin import hamiltonian, eigenvalues
from ansatz import one_particle_ansatz
import pprint
import time
import time

##########################################################################################
j = 2
V = 1
h = hamiltonian(j, V)
h = h[1]
print(h.shape[0])
qc = get_qc("9q-qvm")
eig = smallest_eig_vqe(h, one_particle_ansatz, qc, num_samples=10, opt_algorithm='Nelder-Mead')[0]
print(eig)
##########################################################################################
###### Parameter sweep #####
"""
program = one_particle_ansatz()  # look ma, no arguments!
program.wrap_in_numshots_loop(shots=1000)
executable = qc.compile(program)

thetas = np.linspace(0, 2*np.pi, 21)
results = []
for theta in thetas:
    bitstrings = qc.run(executable, memory_map={'theta': [theta]})
    results.append(np.mean(bitstrings[:, 0]))

# Plot
plt.plot(thetas, results, 'o-')
plt.xlabel(r'$\theta$', fontsize=18)
_ = plt.ylabel(r'$\langle \Psi(\theta) | \frac{1 - Z}{2} | \Psi(\theta) \rangle$', fontsize=18)

# ##### Optimization #####
def objective_function(thetas):
    bitstrings = qc.run(executable, memory_map={'theta': thetas})
    result = np.mean(bitstrings[:, 0])
    return -result


res = minimize(objective_function, x0=[0.1], method='COBYLA')

# Plot
plt.plot(thetas, results, label='scan')
plt.plot([res.x], [-res.fun], '*', ms=20, label='optimization result')
plt.legend()
plt.show()
"""