#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 10:50:49 2019


@author: axelnathanson
"""

# Imports
import numpy as np
from grove.alpha.arbitrary_state.arbitrary_state import create_arbitrary_state



def ucc_ansatz(theta):
    vector = np.zeros(2**(theta.shape[0]))
    vector[[2**i for i in range(theta.shape[0])]] = theta
    return create_arbitrary_state(vector)

