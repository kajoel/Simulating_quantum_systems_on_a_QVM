# -*- coding: utf-8 -*-
"""
Functions for calculating Hamiltonian and corresponding eigenvalues using
the quasi-spin formalism.

Note: currently the code doesn't work for j=0 (corresponding to N=0)

Created on Fri Feb 15 14:16:40 2019
@author: Joel
"""
# Imports
import math
import numpy as np
import scipy.sparse as sparse


"""
Function which returns quasi-spin hamiltonian-matrix for specified value
of j.The function returns two matrices H1 and H2 and the full matrix is
H = H1 \oplus  H2, where \oplus denotes a direct sum
H1 has m \in {-j, -j+2, ...}
H2 has m \in {-j+1, -j+3, ...}

The input parameter e is epsilon in the Lipkin-model
"""
def hamiltonian(j, V, e=1):
    H1 = _quasi_internal(j, -j, V, e)
    H2 = _quasi_internal(j, -j+1, V, e)
    return (H1, H2)


"""
Function which returns eigenvalues for specified value of j. The output is
a 1D ndarray. The eigenvalues are not sorted by size.

The input parameter e is epsilon in the Lipkin-model
"""
def eigenvalues(j, V, e=1):
    eigvals = np.empty(round(2*j+1))
    eigvals[:] = np.NaN
    idx = 0
    for i in range(2):
        H = _quasi_internal(j, -j+i, V, e)
        n = H.shape[0]
        if n > 1:
            eigvals[idx:idx+n-1] = sparse.linalg.eigsh(H, n-1)[0]
            # Since eigsh can only calculate n-1 eigenvalues, use the fact
            # that sum(eigs) = trace(H)
            eigvals[idx+n-1] = np.sum(H.diagonal()) - np.sum(eigvals[idx:idx+n-1])
        else:
            eigvals[idx] = H[0,0]
        idx = idx+n
    return eigvals


# Internal function to keep DRY
def _quasi_internal(j, m_start, V, e):
    # The m value at column/row i is: m_start+2*i.
    size = math.ceil((j-m_start+1)/2)
    m = m_start+2*np.array(range(size),dtype="float")

    idx = np.array(range(size))
    data = np.sqrt((j-m[0:-1])*(j+m[0:-1]+1)*(j-m[0:-1]-1)*(j+m[0:-1]+2))
    J_p2 = sparse.coo_matrix((data, (1+idx[0:-1],idx[0:-1])), shape=(size,size))
    J_p2 = J_p2.tocsr()
    J_m2 = J_p2.transpose()

    J_z = sparse.coo_matrix((m,(idx,idx)))

    H = e*J_z.tocsr() + V/2*(J_p2+J_m2)
    return H
