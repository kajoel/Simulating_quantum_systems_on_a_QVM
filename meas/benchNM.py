#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tus Mar 26 10:06 2019

@author: axel
"""
import numpy as np
from core import ansatz,vqe_eig,init_params
import time

def runtime_NM(H,qc,ansatz_, initial_param,start_samp=100,stop_samp=10000, steps=10):
    samples = np.linspace(start_samp,stop_samp,steps)
    runtime = np.zeros(len(samples))
    if initial_param is None: initial_param = init_params.alternate()
        
    for index,sample in enumerate(samples):
        print(sample)
        sample = int(sample)
        fatol = 16/np.sqrt(sample)
        t1 = time.time()
        vqe_eig.smallest(H,qc,initial_param, ansatz_,samples=sample, fatol=fatol,
                         disp_run_info=True)
        t2 = time.time()
        runtime[index]=t2-t1

    return runtime, samples




    