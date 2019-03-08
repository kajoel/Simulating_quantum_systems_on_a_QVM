#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 16:46:46 2019
matrix_to_hamiltonian:
A function that takes a matrix representation of a Hamiltonian as input and 
generates the Hamiltonian representation with FermionOperators or Jordan-Wigner
transformed to QubitOperators

Input: 
    Matrix: Hamiltonian
    String: JW or jordan to get the matrix transformed. 


hamiltonian_to_pyQuil:
Generates a PauliSum-representation of the Hamiltonian. 

@author: axelnathanson
"""

# Imports
from openfermion.transforms import jordan_wigner
from openfermion.ops import FermionOperator
from scipy.sparse import issparse
# OpenFermion to pyquil
from forestopenfermion import qubitop_to_pyquilpauli as Fermion_to_Pyquil


def matrix_to_pyquil(H):
    return Fermion_to_Pyquil(matrix_to_hamiltonian(H,'JW'))



def matrix_to_hamiltonian(H, transform='none'):    
    Hamiltonian = FermionOperator()
        
    if issparse(H):
        H_sparse = H.tocoo()    
        for i,j,data in zip(H_sparse.row, H_sparse.col, H_sparse.data):
            Hamiltonian += data*FermionOperator( ( (int(i),1) , (int(j),0) ) )
        
    else: 
        for i in range(H.shape[0]):
            for j in range(H.shape[1]):
                Hamiltonian += H[i,j] *FermionOperator( ( (i,1) , (j,0) ) ) 
    
    if(transform == 'jordan' or transform == 'JW'): return jordan_wigner(Hamiltonian)
    else: return Hamiltonian




    

    
    
