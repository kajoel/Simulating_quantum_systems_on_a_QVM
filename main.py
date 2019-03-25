#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 16:12:38 2019

@author: kajoel
"""
from vqe_eig import all, smallest
from lipkin_quasi_spin import hamiltonian, eigs
import pprint
import time
import numpy as np
import pyquil.api as api
from pyquil import get_qc
import time
import ansatz
import init_params

import grove
import pyquil

import misc.compare.dataplotter as dplot

print(grove.__version__)
print(pyquil.__version__)

# qvm = api.QVMConnection()
qc = get_qc('6q-qvm')

j = 1
V = 1
h = hamiltonian(j, V)[0]
print('Hamiltonian: \n:', h)
Realenergies = eigs(j, V)[0]
print('True Eigs: \n', Realenergies)
# TestHamiltonian = H[0].toarray()
# energies = all(TestHamiltonian, one_particle)
start = time.time()

plotter = dplot.dataplotter(nbrlinesperplot=1,nbrfigures=1)

def testprint(x, y):
    plotter.addValues(x[0], y)
    print("Parameter: {}".format(x[0]))
    print("Expectation: {}".format(y))


energies = smallest(h, qc, ansatz.one_particle,
                    initial=init_params.one_particle_ones(h.shape[0]),
                    num_samples=10000, disp_run_info=
                    testprint)[0]
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
    val = smallest(TestHamiltonian, one_particle, qvm, num_samples=10)[0]
    print(val)
    t2 = time.time()
    timer.append(t2-t1)

print('j \t Time elapsed [s]\n')
for index, t in enumerate(timer):
    print('{} \t {}\n'.format(index+1, t))
'''
