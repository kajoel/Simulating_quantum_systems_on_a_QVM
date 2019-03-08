# -*- coding: utf-8 -*-
"""
Script for calculating cj koefficients
cj = irrep multiplicity, related to energy degeneracy

@author: Joel
"""

# Imports:
import numpy as np
import math
from scipy.special import comb as binomial
from collections import Iterable   # to check if N is iterable

# Choose a value for N (specify as int or list of ints):
N = range(11)

# Calculate values for j and cj
j = lambda N: np.array([N/2-k for k in range(0, 1+math.floor(N/2))])
cj = lambda N: np.array([binomial(N, k, exact=True) - binomial(N, k-1, exact=True) for k in range(0, 1+math.floor(N/2))])

# Make sure that N is iterable
if not isinstance(N, Iterable):
    N = [N]
    
# Calculate and print results for all selected N
for n in N:
    # Calulate result
    j_n = j(n)
    cj_n = cj(n)
    # Print value of N
    print("\nN = " + str(n))
    # Check for bad dimension and print result
    if np.sum((2*j_n+1)*cj_n) != 2**n:
        print("The dimensions doesn't add up.")
    else:
        # Print result:
        print("j,\tcj")
        for i in range(j_n.size): print(j_n[i], cj_n[i], sep="\t")
