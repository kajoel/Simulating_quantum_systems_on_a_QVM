#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fre 9 April 17:19 2019

@author: axel
Script to run over night that sweeps through several matrixes, and runs
bays_sample_sweep for every single one. 
"""

# Imports
from core import lipkin_quasi_spin
from meas.bayes_sweep import 


count = 0
for j in range(1,10):
    h_tupple = lipkin_quasi_spin.hamiltonian(j, 1)
    for h in h_tupple:
        if h.shape[0] == 1: continue
        count=multiple_runs_and_save(h, count)


