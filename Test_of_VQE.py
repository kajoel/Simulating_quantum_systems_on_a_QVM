#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mar  10 10:39 2019

@author: axelnathanson
"""
# Imports
import numpy as np
import pyquil.api as api
from scipy.optimize import minimize
from grove.pyvqe.vqe import VQE
import time


# Imports from our projects
from matrix_to_operator import matrix_to_operator_1 
from lipkin_quasi_spin import hamiltonian
from ansatz import one_particle_ansatz as ansatz




qvm = api.QVMConnection()

V = 1

Option = {'disp':True, 'xatol' : 1.0e-2, 'maxiter':10000}
vqe = VQE(minimizer=minimize, minimizer_kwargs={'method': 'Nelder-Mead','options': Option})


timer = []
for j in range(4,5):
    t1 = time.time()
    H,_ = hamiltonian(j,V)
    initial_params = 1/np.sqrt(H.shape[0])*np.array([1 for i in range(H.shape[0])])
    H = matrix_to_operator_1(H)
    
    print('Value of j:{}'.format(j))
    vqe.vqe_run(ansatz, H, initial_params, qvm=qvm)
    t2 = time.time()
    timer.append(t2-t1)

print('j \t Time elapsed [s]\n')
for index,t in enumerate(timer):
    print('{} \t {}\n'.format(index+1,np.round(t,2)))



