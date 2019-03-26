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


def sweep(H, qc,ansatz,matrix_to_opt, num_para=20, start=-10, stop=10, 
                 samples=None):
    """@author: Axel
    Sweeps over parameters for a given Hamiltonian and ansatz. Works for both a 
    2x2 and 3x3 matrix.

    :param H:               Hamiltonian matrix.
    :param qc:              Quantum computer.
    :param ansatz:          Ansatz.
    :param matrix_to_opt:   Method to convert hamiltonian to pyQuil-program.
    :param num_para:        Number of parameters to sweep over.
    :param start:           Value to start the sweep on.
    :param stop:            Value to stop the sweep on.
    :param samples:         Number of samples per parameter.
    
    
    :return: Tuple with expected values and the parameters sweeped over.
    """


    vqe = VQE(minimizer=minimize, minimizer_kwargs={'method': 'Nelder-Mead'})

    if H.shape[0] > 3:
        print('To many parameters to represent in 2 or 3 dimensions')
        return
    elif H.shape[0] is 1:
        print('Nothing to sweep over')
        return
    elif H.shape[0] is 2:
        parameters = np.linspace(start, stop, num_para)
        H = matrix_to_opt(H)
        exp_val = [vqe.expectation(ansatz(np.array([para])), H, samples=samples, 
                                   qc=qc) for para in parameters]
        
        return (exp_val, parameters)
    else:
        H = matrix_to_opt(H)
        exp_val = np.zeros((num_para, num_para))
        mesh_1 = np.zeros((num_para, num_para))
        mesh_2 = np.zeros((num_para, num_para))
        parameters = np.linspace(start, stop, num_para)

        for i, p_1 in enumerate(parameters):
            mesh_1[i] += p_1
            mesh_2[i] = parameters
            
            exp_val[i] = [vqe.expectation(ansatz(np.array([p_1, p_2])), H,
                                          samples=samples, qc=qc)
                                          for p_2 in parameters]
        return (mesh_1,exp_val)

################################################################################
# TESTS
################################################################################




################################################################################
# Main
################################################################################

if __name__ == '__main__':
    print('Hej')
