#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 16:12:38 2019

@author: kajoel
"""
from vqe_eig import calculate_eigenvalues_vqe, smallest_eig_vqe
from lipkin_quasi_spin import hamiltonian, eigenvalues
from ansatz import one_particle_ansatz
import pprint
import time
import numpy as np
import pyquil.api as api
from pyquil import get_qc
import time
from ansatz import carls_initial

import grove
import pyquil
print(grove.__version__)
print(pyquil.__version__)

qvm = api.QVMConnection()
qc = get_qc('6q-qvm')

j = 2
V = 1
H = hamiltonian(j, V)
Realenergies = eigenvalues(j, V)
TestHamiltonian = H[0].toarray()
# energies = calculate_eigenvalues_vqe(TestHamiltonian, one_particle_ansatz)
start = time.time()
energies = smallest_eig_vqe(TestHamiltonian, qc, one_particle_ansatz, num_samples=1)[0]
end = time.time()
pprint.pprint([round(x, 3) for x in Realenergies[0].tolist()])
# pprint.pprint([round(x, 3) for x in sorted(energies)])
pprint.pprint(energies)
print('time: \n:', end - start)

'''
timer = []
for j in range(1, 3):
    t1 = time.time()
    H = hamiltonian(j, V)
    TestHamiltonian = H[1]
    val = smallest_eig_vqe(TestHamiltonian, one_particle_ansatz, qvm, num_samples=10)[0]
    print(val)
    t2 = time.time()
    timer.append(t2-t1)

print('j \t Time elapsed [s]\n')
for index, t in enumerate(timer):
    print('{} \t {}\n'.format(index+1, t))
'''
