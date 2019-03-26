#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tus Mar 26 10:06 2019

@author: axel
"""

from core import init_params,matrix_to_op,ansatz,lipkin_quasi_spin, init_params
from pyquil import get_qc
from meas import sweep, benchNM
import matplotlib.pyplot as plt
import time


def test_1():
    j,V = 1,1

    H = lipkin_quasi_spin.hamiltonian(j,V)[0]
    qubits = H.shape[0]
    qc = get_qc('{}q-qvm'.format(qubits))
    ans = ansatz.one_particle
    ma_to_op = matrix_to_op.one_particle
    exp, para = sweep.sweep(H,qc,ans,ma_to_op,samples=1000, num_para=50,start=-2,stop=2)


    plt.plot(para,exp, label='One')


def test_2():
    j,V = 1,1


    H = lipkin_quasi_spin.hamiltonian(j,V)[0]
    qubits = int.bit_length(H.shape[0])

    qc = get_qc('{}q-qvm'.format(qubits))
    ans = ansatz.multi_particle
    ma_to_op = matrix_to_op.multi_particle
    exp, para = sweep.sweep(H,qc,ans,ma_to_op,samples=1000, num_para=50,start=-2,stop=2)

    plt.plot(para,exp, label='Multi')

def test_bench_NM():
    j,V = 1,1
    h = lipkin_quasi_spin.hamiltonian(j,V)[0]
    H = matrix_to_op.one_particle(h)
    qubits = h.shape[0]
    qc = get_qc('{}q-qvm'.format(qubits))
    ansatz_ = ansatz.one_particle

    runtime, samples = benchNM.runtime_NM(H,qc,ansatz_, init_params.alternate(h.shape[0]), 
                                          start_samp=2000, stop_samp=10000,steps=40)

    ansatz_=ansatz.multi_particle
    runtime2, samples2 = benchNM.runtime_NM(H,qc,ansatz_, init_params.alternate(h.shape[0]), 
                                          start_samp=2000, stop_samp=10000,steps=40)

    plt.plot(samples,runtime,label='One Particle ansatz')
    plt.plot(samples2,runtime2,label='Multi Particle ansatz')
    plt.legend()
    plt.show()






if __name__ == '__main__':
    t = time.time()
    test_1()
    t1=time.time()
    test_2()
    t2= time.time()
    plt.legend()
    plt.show()
    


