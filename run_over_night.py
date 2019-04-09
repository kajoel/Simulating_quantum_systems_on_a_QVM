#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tus Mar 27 14:26 2019

@author: axel
Script to run over night that sweeps through several matrixes, and runs
NM_sample_sweepfor every single one. 
"""

# Imports
from core import lipkin_quasi_spin
from meas.nm_sample_sweep import multiple_runs_and_save


count = 0
for j in range(1,10):
    h_tupple = lipkin_quasi_spin.hamiltonian(j, 1)
    for h in h_tupple:
        if h.shape[0] == 1: continue
        count=multiple_runs_and_save(h, count)


