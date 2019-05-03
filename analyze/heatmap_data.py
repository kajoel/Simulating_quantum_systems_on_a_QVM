# @author = Carl, 1/5
# Imports
import numpy as np
from constants import ROOT_DIR
from os.path import join
from core import data
from analyze import NM_fel_measmax

def NM_save(version, size, ansatz_name, minimizer):
    base_dir = join(ROOT_DIR, f'data/final_nm/v{version}')
    data_file = f'{ansatz_name}_{minimizer}_size={size}.pkl'

    data_, _ = data.load(data_file, base_dir)

    fel = []
    samples_lst = np.array([])
    max_meas = np.linspace(1e6, 3e6, 41)
    nr = []

    for i, y in enumerate(data_):
        print(f'Klar med {i+1}/{len(data_)}')

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

    file = f'NM_heatmap/v{version}/{ansatz_name}_{minimizer}_size={size}.pkl'
    data2 = max_meas, samples_lst, fel
    data.save(file, data2, extract=True)


def bayes_save(version, size, ansatz_name, minimizer):
    base_dir = join(ROOT_DIR, f'data/final_bayes/v{version}')
    data_file = f'{ansatz_name}_{minimizer}_size={size}.pkl'

    data_, _ = data.load(data_file, base_dir)

    #identifier =size, ansatz_name, minimizer, repeats, hamiltonian_idx, \
    #            int(max_meas),int(samples))

    #for max_meas in np.linspace(1e6, 3e6, 41):
    #for samples in np.linspace(1e4, 3e5, 41):

    fel = np.zeros([41, 41])
    samples_lst = []
    max_meas_lst = []
    nr = np.zeros([41, 41])

    for i, y in enumerate(data_):
        identifier, result = y
        samples = identifier[6]
        max_meas = identifier[5]
        arr = np.asarray(np.abs(np.linspace(1e4, 3e5, 41) - samples))
        samples_idx = np.argmin(arr)
        max_meas_idx = np.argmin(np.abs(np.linspace(1e6, 3e6, 41) - max_meas))

        eig = result['correct']
        fun_none = result['fun_none']

        error = (eig - fun_none) / eig * 100

        max_meas_lst.append(max_meas)
        samples_lst.append(samples)

        fel[samples_idx][max_meas_idx] +=error
        nr[samples_idx][max_meas_idx] +=1

    for j in range(41):
        for k in range(41):
            if np.any(nr[j,k] != 0):
                fel[j,k]/=nr[j,k]
            else:
                fel[j,k] = 5

        print(fel.shape[0])
    file = f'NM_heatmap/v{version}/{ansatz_name}_{minimizer}_size={size}.pkl'
    data2 = np.linspace(1e6, 3e6, 41), np.linspace(1e4, 3e5, 41), fel
    data.save(file, data2, extract=True)


version = 3
size = 3
ansatz_name = 'multi_particle'
minimizer = 'bayes'

bayes_save(version, size, ansatz_name, minimizer)