#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 16:12:38 2019

@author: kajoel
"""
from vqe_eig import calculate_eigenvalues_vqe
from lipkin_quasi_spin import hamiltonian, eigenvalues
from ansatz import one_particle_ansatz
import pprint
import numpy as np

j = 10/2
V = 1
H = hamiltonian(j, V)
Realenergies = eigenvalues(j, V)
TestHamiltonian = H[1].toarray()
energies = calculate_eigenvalues_vqe(TestHamiltonian, one_particle_ansatz)
pprint.pprint([round(x, 3) for x in Realenergies[1].tolist()])
pprint.pprint([round(x, 3) for x in sorted(energies)])
