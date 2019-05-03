"""
Created on Fri Mar  8 16:12:38 2019

@author: kajoel
"""
from core import vqe_eig, vqe_override, matrix_to_op
from core import lipkin_quasi_spin
import pprint
from pyquil import get_qc
import time
from core import ansatz
from core import init_params
from scipy.optimize import minimize
from core import callback as cb
import grove
import pyquil

print(grove.__version__)
print(pyquil.__version__)

j = 1
V = 1
h = lipkin_quasi_spin.hamiltonian(j, V)[0]
print('Hamiltonian: \n:', h)
Realenergies = lipkin_quasi_spin.eigs(j, V)[0]
print('True Eigs: \n', Realenergies)

qc = get_qc(str(int.bit_length(h.shape[0])) + 'q-qvm')
H = matrix_to_op.multi_particle(h)
ansatz_ = ansatz.multi_particle(h)
initial_params = init_params.alternate(h.shape[0])

display_after_run = True
xatol = 1e-2
fatol = 1e-2
maxiter = 10000
opt_algorithm = 'Nelder-Mead'
disp_options = {'disp': display_after_run, 'xatol': xatol, 'fatol': fatol,
                'maxiter': maxiter}

vqe = vqe_override.VQE_override(minimizer=minimize,
                                minimizer_kwargs={'method':
                                                      opt_algorithm,
                                                  'options': disp_options})
start = time.time()
energies = vqe_eig.smallest(H, qc, initial_params, vqe,
                            ansatz_=ansatz_,
                            samples=200, disp=True,
                            callback=cb.restart(5, 1e-3,
                                                disp=True),
                            attempts=5)['fun']
end = time.time()
pprint.pprint([round(x, 3) for x in Realenergies.tolist()])
# pprint.pprint([round(x, 3) for x in sorted(energies)])
pprint.pprint("energy: " + str(energies))
print('time: \n:', end - start)

'''
timer = []
for j in range(1, 3):
    t1 = time.time()
    H = hamiltonian(j, V)
    TestHamiltonian = H[1]
    val = vqe_eig.smallest(TestHamiltonian, one_particle, qvm, 
    num_samples=10)[0]
    print(val)
    t2 = time.time()
    timer.append(t2-t1)

print('j \t Time elapsed [s]\n')
for index, t in enumerate(timer):
    print('{} \t {}\n'.format(index+1, t))
'''
