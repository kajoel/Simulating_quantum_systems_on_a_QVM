#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Tid 16 April 14:55 2019

'''

from core import init_params,matrix_to_op,ansatz,lipkin_quasi_spin, data, vqe_eig
from pyquil import get_qc
import matplotlib.pyplot as plt
import numpy as np

def create_qcs(h):
    qc_one = get_qc('{}q-qvm'.format(h.shape[0]))
    qc_multi = get_qc('{}q-qvm'.format(int.bit_length(h.shape[0])))
    return qc_one, qc_multi
    
    


def intervall_sweep(j_max, V=1, samples = 500):
    intervall = range(1,15,2)
    j_range = range(1,j_max)
    print(j_range)
    data = np.zeros( (len(j_range), len(intervall)) ) 
    for j in j_range:
        h = lipkin_quasi_spin.hamiltonian(j, V)[0]
        ansatz_ = [ansatz.one_particle_ucc(h.shape[0]),
                   ansatz.multi_particle(h.shape[0])]
        _, qc_multi = create_qcs(h)

        H = matrix_to_op.multi_particle(h)
        facit= vqe_eig.smallest(H, qc_multi, init_params.alternate(h.shape[0]),
                                ansatz_[1])
        
        for i,inter in enumerate(intervall):
            dimension = [(-inter, inter) for i in range(h.shape[0]-1)]
        
            temp =  [vqe_eig.smallest_bayes(H, qc_multi, dimension, ansatz_[1], 
                     samples=samples, n_random_starts=5, n_calls=10, 
                     return_all_data=True, 
                     disp=False)['x'] for i in range(3)]
            data[j-1, i] = np.linalg.norm((facit[1] - np.mean(temp)))
    

    from pandas import DataFrame
    import seaborn as sns
    j_range = [j+1 for j in j_range]
    df = DataFrame(data, index=j_range, 
                   columns=intervall)
    plt.figure()
    sns.heatmap(df)        
    



        


if __name__ == '__main__':
    intervall_sweep(6)
    plt.show()
