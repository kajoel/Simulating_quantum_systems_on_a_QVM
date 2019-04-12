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
from os.path import join

ansatzer = [ansatz.multi_particle, ansatz.one_particle_ucc, 
            ansatz.one_particle]
convert_op = [matrix_to_op.multi_particle, matrix_to_op.one_particle,
              matrix_to_op.one_particle]
V = 1

for j in range(2,5):
    print('Starting with j={}'.format(j))
    h_tupple = lipkin_quasi_spin.hamiltonian(j, V)
    for i,h in enumerate(reversed(h_tupple)):
        if h.shape[0] == 1: continue
        samples_stop = 3000*(h.shape[0]-1)
        func_evals = 30*(h.shape[0]-1)
        if i==0: i=1
        else: i=0
        print(i)
        if j is 2 and i is 1: continue
        for index,ansatz_ in enumerate(ansatzer):
            if index < 2 and i == 0 and j == 2: continue
            file_name = join('heatmapsBayes',
                             'run3_updatedSampleDef_{}_j{}V{}i{}'.format(ansatz_.__name__,j,V,i))    
            heatmap(ansatz_, convert_op[index], h, save=True, 
                    file_name=file_name,
                    sample_step=10, sample_start=100, sample_stop=samples_stop, 
                    func_steps=10, func_start=10, func_stop=func_evals,
                    measurments=4, plot_after_run=False)




