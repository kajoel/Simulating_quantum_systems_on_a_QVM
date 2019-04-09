#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri April 5 11:06 2019

@author: axel
"""

from core import init_params,matrix_to_op,ansatz,lipkin_quasi_spin, data, vqe_eig
from pyquil import get_qc
import matplotlib.pyplot as plt

def run_bayes_opt():
    j,V = 2,1

    h = lipkin_quasi_spin.hamiltonian(j,V)[1]
    print(h)
    print(lipkin_quasi_spin.eigs(j,V))
    qubits = h.shape[0]
    qc = get_qc('{}q-qvm'.format(qubits))
    ansatz_ = ansatz.multi_particle(h.shape[0])
    H = matrix_to_op.multi_particle(h)
    dimension = [(-1.0, 1.0) for i in range(h.shape[0]-1)]
    print(dimension)



    eig_val = vqe_eig.smallest_bayes(H, qc, dimension, ansatz_, samples=100, 
                           n_random_starts=4, n_calls=20, return_all_data=True)
    

    initial_p = init_params.alternate(h.shape[0])
    vqe_eig.smallest(H, qc, initial_p, ansatz_, return_all_data=True)
    print(eig_val['fun'])

    for key in eig_val: print(key)
    print(eig_val['expectation_vals'])
    print(eig_val['expectation_vars'])
    print(len(eig_val['iteration_params']))

    


if __name__ == '__main__':
    run_bayes_opt()
