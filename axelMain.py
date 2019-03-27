#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tus Mar 26 10:06 2019

@author: axel
"""

from core import init_params,matrix_to_op,ansatz,lipkin_quasi_spin, init_params,data
from pyquil import get_qc
from meas import sweep, benchNM
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time


def sweep_one():
    j,V = 1,1

    H = lipkin_quasi_spin.hamiltonian(j,V)[0]
    qubits = H.shape[0]
    qc = get_qc('{}q-qvm'.format(qubits))
    ans = ansatz.one_particle
    ma_to_op = matrix_to_op.one_particle
    exp, para = sweep.sweep(H,qc,ans,ma_to_op,samples=1000, num_para=50,start=-2,stop=2)


    plt.plot(para,exp, label='One')


def sweep_multi():
    j,V = 1,1


    H = lipkin_quasi_spin.hamiltonian(j,V)[0]
    qubits = int.bit_length(H.shape[0])

    qc = get_qc('{}q-qvm'.format(qubits))
    ans = ansatz.multi_particle
    ma_to_op = matrix_to_op.multi_particle
    exp, para = sweep.sweep(H,qc,ans,ma_to_op,samples=1000, num_para=50,start=-2,stop=2)

    plt.plot(para,exp, label='Multi')


def sweep_2Dim(ansatz_,h=None, j=2,V=1,samples_=1000, save = False):
    if h is None: h = lipkin_quasi_spin.hamiltonian(j,V)[0]
    
    if h.shape[0] !=3:
        print('Only 3x3 dimension hamiltonians supported, not {}x{}'.format(h.shape[0],h.shape[0]))
        return
    
    if ansatz_ is ansatz.one_particle:
        matrix_operator = matrix_to_op.one_particle
        qubits = h.shape[0]
    elif ansatz_ is ansatz.multi_particle:
        matrix_operator = matrix_to_op.multi_particle
        qubits = int.bit_length(h.shape[0])
    else:
        print('Unknown ansatz')
        return
    qc = get_qc('{}q-qvm'.format(qubits))

    exp, para1,para2 = sweep.sweep(h,qc,ansatz_,matrix_operator,samples=samples_,
                                   num_para=100,start=-3,stop=3)


    fig = plt.figure(0)
    ax = fig.add_subplot(111, projection='3d')
    # Plot the surface
       
    ax.plot_surface(para1, para2, exp)
    
    if save:
        metadata = {'ansatz': ansatz_, 'Hamiltonian': h, 'samples': None,'minimizer': 'Nelder-Mead',
                    'matrix_to_op': matrix_operator, 
                    'Type of measurement': 'Sweep over parameters in 2 dimensions, from -3 to 3 with 100 steps.'}

        data_to_save = (exp,para1,para2)

        data.save('2DimSweepSampNone',data_to_save,metadata)


def test_bench_NM():
    j,V = 1,1
    h = lipkin_quasi_spin.hamiltonian(j,V)[0]
    H = matrix_to_op.one_particle(h)
    qubits = h.shape[0]
    qc = get_qc('{}q-qvm'.format(qubits))
    ansatz_ = ansatz.one_particle

    runtime, samples = benchNM.runtime_NM(H,qc,ansatz_, init_params.alternate(h.shape[0]), 
                                          start_samp=2000, stop_samp=10000,steps=20)

    ansatz_=ansatz.multi_particle
    runtime2, samples2 = benchNM.runtime_NM(H,qc,ansatz_, init_params.alternate(h.shape[0]), 
                                          start_samp=2000, stop_samp=8000,steps=20)

    plt.plot(samples,runtime,label='One Particle ansatz')
    plt.plot(samples2,runtime2,label='Multi Particle ansatz')
    plt.legend()
    plt.show()


def save_sweep():
    j,V = 1,1

    h = lipkin_quasi_spin.hamiltonian(j,V)[0]
    qubits = h.shape[0]
    qc = get_qc('{}q-qvm'.format(qubits))
    ansatz_ = ansatz.one_particle
    ma_to_op = matrix_to_op.one_particle
    exp, para = sweep.sweep(h,qc,ansatz_,ma_to_op,samples=None, num_para=5000,start=-3,stop=3)

    plt.plot(para,exp, label='One particle, sample None')

    metadata = {'ansatz': ansatz_, 'Hamiltonian': h, 'samples': None,'minimizer': 'Nelder-Mead',
                'matrix_to_op': ma_to_op, 
                'Type of measurement': 'Sweep over parameters, from -3 to 3 with 5000 steps.'}

    data_to_save = (exp,para)

    data.save('1DimSweepSampNone',data_to_save,metadata)



if __name__ == '__main__':
    ansatz_=ansatz.one_particle
    sweep_one()
    plt.show()
    

