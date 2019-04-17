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
import time

def run_bayes_opt(n_jobs=1):
    j,V = 3,1

    h = lipkin_quasi_spin.hamiltonian(j,V)[1]
    qubits = h.shape[0]
    qc = get_qc('{}q-qvm'.format(qubits))
    ansatz_ = ansatz.multi_particle(h)
    H = matrix_to_op.multi_particle(h)
    dimension = [(-20.0, 20.0) for i in range(h.shape[0]-1)]
    samples = 1000

    t1 = time.time()
    temp =  vqe_eig.smallest_bayes(H, qc, dimension, ansatz_, samples=samples, 
                           n_random_starts=5, n_calls=10, return_all_data=True, 
                           disp=False, n_jobs=n_jobs)['x']
    t2 = time.time()

    print(temp)
    print('Eloped time for {} jobs: {}'.format(n_jobs, t2-t1))


if __name__ == '__main__':
    for jobs in range(1,10):
        run_bayes_opt(jobs)
