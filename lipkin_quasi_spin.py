# -*- coding: utf-8 -*-
"""
Functions for calculating Hamiltonian and corresponding eigenvalues using
the quasi-spin formalism.

Note: - Currently the code doesn't work for j=0 (corresponding to N=0).
    - eigsh can returns eigenvalues and -vectors so this code could be used
      for the latter as well (with minimal modification)

Created on Fri Feb 15 14:16:40 2019
@author: Joel
"""
# Imports
import math
import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import eigsh
from functools import lru_cache


"""
Function which returns quasi-spin hamiltonian-matrix for specified value
of j.The function returns two matrices H1 and H2 and the full matrix is
H = H1 \oplus  H2, where \oplus denotes a direct sum
H1 has m \in {-j, -j+2, ...}
H2 has m \in {-j+1, -j+3, ...}

The input parameter e is epsilon in the Lipkin-model
"""
@lru_cache(maxsize=1)
def hamiltonian(j, V, e=1):
    H1 = _quasi_internal(j, -j, V, e)
    H2 = _quasi_internal(j, -j+1, V, e)
    return (H1, H2)


"""
Function which returns eigenvalues for specified value of j. The output is
two 1D ndarrays with sorted eigenvalues corresponding to H1 and H2.

The input parameter e is epsilon in the Lipkin-model
"""
def eigenvalues(j, V, e=1):
    # Preallocate and store in tuple to keep DRY
    eigvals= (np.empty(math.ceil((2*j+1)/2)), np.empty(math.floor((2*j+1)/2)))

    # Get matrices
    args = [j, V]
    args.extend([e] if e!=1 else [])  # to get cache to work with default e
    H = hamiltonian(*args)

    for i in range(2):
        # eigsh can only calculate all but one eigenvalue, so the last one is
        # calculated using sum(eigs) = trace(H).
        if H[i].shape[0] > 1:
            eigvals[i][0:-1] = eigsh(H[i], H[i].shape[0]-1)[0]
            eigvals[i][-1] = np.sum(H[i].diagonal()) \
                             - np.sum(eigvals[i][0:-1])
        else:
            eigvals[i][-1] = H[i][0,0]
        eigvals[i].sort()
    return eigvals


"""
Function which returns eigenvalues for specified value of j. The output is
a 1D ndarray. The eigenvalues are sorted by size and only the positive
eigenvalues are returned.

The input parameter e is epsilon in the Lipkin-model
"""
def eigenvalues_positive(j, V, e=1):
    # We use that we now that there are floor((2j+1)/2) strictly positive
    # eigenvalues. This avoids numerical problems with eigenvalues
    # that are 0 but calculated to (say) 1.7e-15
    eigvals = eigenvalues(j, V, e)
    eigvals = np.concatenate(eigvals)
    eigvals.sort()
    return eigvals[math.ceil((eigvals.shape[0])/2):]


# Internal function to keep DRY
def _quasi_internal(j, m_start, V, e):
    # The m value at column/row i is: m_start+2*i.
    size = math.ceil((j-m_start+1)/2)
    m = m_start+2*np.array(range(size),dtype="float")

    idx = np.array(range(size))
    data = np.sqrt((j-m[0:-1])*(j+m[0:-1]+1)*(j-m[0:-1]-1)*(j+m[0:-1]+2))
    J_p2 = sparse.coo_matrix((data,(1+idx[0:-1],idx[0:-1])),shape=(size,size))
    J_p2 = J_p2.tocsr()
    J_m2 = J_p2.transpose()

    J_z = sparse.coo_matrix((m,(idx,idx)))

    H = e*J_z.tocsr() + V/2*(J_p2+J_m2)
    return H
