#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri April 5 11:06 2019

@author: axel
"""

from core import init_params,matrix_to_op,ansatz,lipkin_quasi_spin, data, vqe_eig
from pyquil import get_qc
import matplotlib.pyplot as plt
import numpy as np

def run_bayes_opt():
    j,V = 2,1

    h = lipkin_quasi_spin.hamiltonian(j,V)[1]
    qubits = h.shape[0]
    qc = get_qc('{}q-qvm'.format(qubits))
    ansatz_ = ansatz.multi_particle(h.shape[0])
    H = matrix_to_op.multi_particle(h)
    dimension = [(-20.0, 20.0) for i in range(h.shape[0]-1)]
    print(dimension)
    samples = 1000

    
    temp =  [vqe_eig.smallest_bayes(H, qc, dimension, ansatz_, samples=samples, 
                           n_random_starts=5, n_calls=10, return_all_data=True, 
                           disp=False)['x'] for i in range(10)]
    print(np.mean(temp))


if __name__ == '__main__':
    [run_bayes_opt() for i in range(5)]
