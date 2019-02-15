# -*- coding: utf-8 -*-
"""
Script for calculating cj koefficients
cj = irrep multiplicity, related to energy degeneracy

TODO: rewrite as function that can be called with multiple values for N at once

NOTE: this works for (at least) N<30 but doesn't work for N>50, consider
        changing scipy.special.binom to scipy.special.comb and use
        exact=True (this will slow the program down but give correct results
        even for larger N)
"""

# Imports:
import numpy as np
import math
from scipy.special import binom as binomial
import sys

# Choose a value for N:
N = 30

# Calculate values for j and cj
j = np.array([N/2-k for k in range(0, 1+math.floor(N/2))])
cj = np.array([binomial(N, k) - binomial(N, k-1) for k in range(0, 1+math.floor(N/2))])

# Check for bad dimension
if np.sum((2*j+1)*cj) != 2**N:
    print("The dimensions doesn't add up.")
    sys.exit()

# Print result:
print("j,\tcj")
for i in range(j.size): print(j[i], cj[i], sep="\t")