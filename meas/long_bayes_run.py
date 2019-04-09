#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fre 9 April 17:19 2019

@author: axel
Script to run over night that sweeps through several matrixes, and runs
bays_sample_sweep for every single one. 
"""

# Imports
from core import ansatz, matrix_to_op, lipkin_quasi_spin
from meas.bayes_sweep import heatmap
import matplotlib.pyplot as plt

ansatzer = [ansatz.multi_particle, ansatz.one_particle_ucc, 
            ansatz.one_particle]
convert_op = [matrix_to_op.multi_particle, matrix_to_op.one_particle,
              matrix_to_op.one_particle]

j, V, i = 1, 1, 0
h = lipkin_quasi_spin.hamiltonian(1, 1)[0]

for i,ansatz_ in enumerate(ansatzer):
        heatmap(ansatz_, convert_op[i], h, save=True, 
                file_name='heatmap{}j{}V{}i{}'.format(ansatz_.__name__,j,V,i),
                sample_step=2, sample_start=100, sample_stop=2000, 
                func_steps=2, func_start=10, func_stop=100, 
                measurments=2, plot_after_run=False)

plt.show()



