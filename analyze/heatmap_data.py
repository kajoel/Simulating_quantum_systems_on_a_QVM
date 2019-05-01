# @author = Carl, 1/5
# Imports
import numpy as np
import matplotlib.pyplot as plt
import itertools
from constants import ROOT_DIR
from os.path import join

from core import data
from analyze import NM_fel_measmax
from plot import tikzfigure

version = 3
size = 3
ansatz_name = 'multi_particle'
minimizer = 'nelder-mead'

base_dir = join(ROOT_DIR, f'data/final_nm/v{version}')
file = f'{ansatz_name}_{minimizer}_size={size}.pkl'

data_, _ = data.load(file, base_dir)

#fel = np.empty((0,41), float)
#fel = np.array([])
fel = []
samples_lst = np.array([])

max_meas = np.linspace(1e6, 3e6, 41)

nr = []
for i, y in enumerate(data_):
    print(f'Klar med {i}/{len(data_)}')

    identifier, result = y
    samples = identifier[6]

    x = result['iteration_params_all']
    fun = result['expectation_vals_all']
    eig = result['correct']
    fun_evals = []

    for max_meas_ in max_meas:
        if max_meas_ % samples == 0:
            fun_evals.append(int(max_meas_ / samples) - 1)
        else:
            fun_evals.append(int(max_meas_ / samples))

    fel_none = NM_fel_measmax.fel_measmax(x, fun, identifier, fun_evals)

    error = (eig - fel_none) / eig * 100

    if samples not in samples_lst:
        samples_lst = np.append(samples_lst, samples)
        #fel = np.append(fel, error, axis=0)
        #fel = np.vstack((fel, error))
        fel.append(error)
        nr.append(1)
    else:
        idx = np.argmin(samples_lst - samples)
        fel[idx] = (fel[idx] + error)
        nr[idx] +=1

fel = np.array(fel)

for i, rows in enumerate(fel):
    fel[i,:] = rows/nr[i]

sort_idx = np.argsort(samples_lst)

samples_lst = samples_lst[sort_idx]
fel = fel[sort_idx,:]

file = 'NM_heatmap/v1.pkl'
data2 = max_meas, samples_lst, fel
data.save(file, data2, extract=True)
