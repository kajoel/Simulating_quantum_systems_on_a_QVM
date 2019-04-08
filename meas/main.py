"""
Created on Fri Mar  8 16:12:38 2019

@author: kajoel
"""
from core import vqe_eig
from core import lipkin_quasi_spin
import pprint
from pyquil import get_qc
import time
from core import ansatz
from core import init_params

import grove
import pyquil

print(grove.__version__)
print(pyquil.__version__)

# qvm = api.QVMConnection()
qc = get_qc('6q-qvm')

j = 1
V = 1
h = lipkin_quasi_spin.hamiltonian(j, V)[0]
print('Hamiltonian: \n:', h)
Realenergies = lipkin_quasi_spin.eigs(j, V)[0]
print('True Eigs: \n', Realenergies)
# TestHamiltonian = H[0].toarray()
# energies = all(TestHamiltonian, one_particle)
start = time.time()
energies = vqe_eig.smallest(h, qc, init_params.ones(h.shape[0]),
                            ansatz_=ansatz.one_particle,
                            samples=None, disp=
                            True, display_after_run=True)[0]
end = time.time()
pprint.pprint([round(x, 3) for x in Realenergies.tolist()])
# pprint.pprint([round(x, 3) for x in sorted(energies)])
pprint.pprint(energies)
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
