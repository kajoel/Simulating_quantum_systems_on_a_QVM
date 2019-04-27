# Carl 14/4
###############################################################################
# Imports
from core import data
import numpy as np
import matplotlib.pyplot as plt
from constants import ROOT_DIR
from os.path import join
import itertools
from core import ansatz
from core.lipkin_quasi_spin import hamiltonian
from core import init_params

###############################################################################

base_dir = join(ROOT_DIR, 'data/NelderMead_Restart_v2')

j, V, matrix = 1, 1, 0
ansatz_types = ['one_particle_ucc', 'multi_particle']

h = hamiltonian(j, V)[matrix]
dim = h.shape[0]

data_ = {}


H1 = ansatz.ansatz_type(ansatz_types[0], h, dim)[0]
H2 = ansatz.ansatz_type(ansatz_types[1], h, dim)[0]

xatol = 1e-2
tol_para = 1e-2
max_iter = 20
increase_samples = 0

parameters = {'xatol': xatol, 'tol_para': tol_para, 'max_iter': max_iter,
              'increase_samples': increase_samples}

metadata = {'info': 'NelderMead Restart', 'j': j, 'V': V, 'matrix': matrix,
            'ansatz': ansatz_types, 'initial_params': [init_params.ucc(dim), \
            init_params.alternate(dim)], 'H': [H1, H2], 'parameters': parameters}



for ansatz_name in ansatz_types:

    data_tmp4 = []
    H = ansatz.ansatz_type(ansatz_name, h, dim)[0]
    samples = np.linspace(500, len(H) * 10000, 100)

    for sample in samples:
        data_tmp3 = {}
        sample = int(round(sample))

        for max_para in range(5, 10):
            data_tmp2 = {}
            file = 'NelderMead_Restart_j={}_samples={}_ansatz={}_maxpara={}.pkl'. \
            format(j, sample, ansatz_name, max_para)

            try:
                data_tmp, metadata_tmp = data.load(file, base_dir)
            except FileNotFoundError:
                break


            for iter in range(5):
                data_tmp5 = data_tmp[iter]
                data_tmp5['len_H'] = metadata_tmp['len_H']
                data_tmp5['fatol'] = metadata_tmp['paramaters']['fatol']
                data_tmp2['iter={}'.format(iter+1)] = data_tmp5


            data_tmp3['max_param={}'.format(max_para)] = data_tmp2

        if data_tmp3 != {}:
            data_tmp4.append(data_tmp3)

    data_[ansatz_name] = data_tmp4




file = 'NMR_j={}.pkl'.format(j)

data.save(file, data_, metadata, base_dir)