#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri April 5 11:06 2019

@author: axel
"""
import core.interface
from core import init_params,matrix_to_op,ansatz,lipkin_quasi_spin, data, vqe_eig, create_vqe
from pyquil import get_qc
import matplotlib.pyplot as plt
import numpy as np
import time

def run_bayes_opt(n_jobs=1):
    j,V = 2,1

    h = lipkin_quasi_spin.hamiltonian(j,V)[1]
    qubits = h.shape[0]
    qc = get_qc('{}q-qvm'.format(qubits))
    vqe = core.interface.default_bayes(n_calls=15)
    ansatz_ = ansatz.multi_particle(h)
    H = matrix_to_op.multi_particle(h)
    dimension = [(-20.0, 20.0) for i in range(h.shape[0]-1)]
    samples = 2000

    t1 = time.time()
    temp =  vqe_eig.smallest(H, qc, dimension, vqe, ansatz_, samples=samples, 
                             return_all=True)
    
    for key in temp: 
        print(key)
        print(type(temp[key]))
    
    print(len(temp['iteration_params']))
    print(temp['iteration_params'][-1])
    print(type(temp['iteration_params'][-1]))
    print(temp['iteration_params'][-1].x_iters)

    t2 = time.time()


    print('Eloped time for {} jobs: {}'.format(n_jobs, t2-t1))
    print('Number of evaluations on qc:{}'.format(samples*30))


if __name__ == '__main__':
    for jobs in range(1,2):
        run_bayes_opt(jobs)
