# -*- coding: utf-8 -*-
"""
Function which returns quasi-spin hamiltonian-matrix for specified value of j
The function will return two matrices H1 and H2 and the full matrix is
H = H1 \oplus  H2, where \oplus denotes a direct sum
H1 has m \in {-j, -j+2, ...}
H2 has m \in {-j+1, -j+3, ...}

Created on Fri Feb 15 14:16:40 2019
@author: Joel
"""
# Imports
import math
import numpy as np
import scipy.sparse as sparse

# Main function
def quasi_spin_hamiltonian(j, V, e=1):
    H1 = _quasi_internal(j, -j, V, e)
    H2 = _quasi_internal(j, -j+1, V, e)
    return (H1, H2)

# Internal function to avoid duplication
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
