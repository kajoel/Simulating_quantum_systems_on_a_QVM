#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mar  10 10:39 2019

@author: axelnathanson
"""
# Imports
import numpy as np
from scipy.optimize import minimize
from grove.pyvqe.vqe import VQE

# Imports to run test
from core import lipkin_quasi_spin, matrix_to_op, ansatz
from pyquil import get_qc
import matplotlib.pyplot as plt
from pandas import DataFrame
import seaborn as sns



def sweep(h, qc, ansatz_, matrix_operator, num_para=20, start=-10, stop=10,
                 samples=None):
    """@author: Axel
    Sweeps over parameters for a given Hamiltonian and ansatz. Works for both a 
    2x2 and 3x3 matrix.

    :param h:               Hamiltonian matrix.
    :param qc:              Quantum computer.
    :param ansatz:          Ansatz.
    :param matrix_operator: Method to convert hamiltonian to pyQuil-program.
    :param num_para:        Number of parameters to sweep over.
    :param start:           Value to start the sweep on.
    :param stop:            Value to stop the sweep on.
    :param samples:         Number of samples per parameter.
    
    
    :return: Tuple with: expected values and the parameters sweeped over.
    """

    vqe = VQE(minimizer=minimize, minimizer_kwargs={'method': 'Nelder-Mead'})

    if h.shape[0] > 3:
        print('To many parameters to represent in 2 or 3 dimensions')
        return
    elif h.shape[0] is 1:
        print('Nothing to sweep over')
        return
    elif h.shape[0] is 2:
        parameters = np.linspace(start, stop, num_para)
        H = matrix_operator(h)
        exp_val = [vqe.expectation(ansatz_(np.array([para])), H, samples=samples, 
                                   qc=qc) for para in parameters]
        return exp_val, parameters
    else:
        H = matrix_operator(h)
        exp_val = np.zeros((num_para, num_para))
        mesh_1 = np.zeros((num_para, num_para))
        mesh_2 = np.zeros((num_para, num_para))
        parameters = np.linspace(start, stop, num_para)

        for i, p_1 in enumerate(parameters):
            mesh_1[i] += p_1
            mesh_2[i] = parameters

            exp_val[i] = [vqe.expectation(ansatz_(np.array([p_1, p_2])), H,
                                          samples=samples, qc=qc)
                          for p_2 in parameters]
            print('Done with sweep number {}/{}'.format(i+1,len(parameters)))
        return (exp_val,parameters)


################################################################################
# TESTS
################################################################################


################################################################################
# Main
################################################################################

if __name__ == '__main__':
    h = lipkin_quasi_spin.hamiltonian(2, 1)[0]
    
    convert_op = matrix_to_op.one_particle
    ansatz_ = ansatz.one_particle(h.shape[0])

    if convert_op is matrix_to_op.one_particle: qubit = h.shape[0]
    elif convert_op is matrix_to_op.multi_particle:
        qubit = int.bit_length(h.shape[0])
    


    qc = get_qc('{}q-qvm'.format(qubit))
    exp_val, parameters =sweep(h,qc,ansatz_, convert_op, 50, -2, 2)
    df = DataFrame(exp_val, index=parameters, columns=parameters)
    sns.heatmap(df)

    plt.show()
