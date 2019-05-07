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

    data_= data.load(data_file, base_dir)[0]

    fel = []
    samples_lst = []
    max_meas = np.linspace(50000, 3e6, 60)
    nr = []
    identifier_set = set()

    for i, y in enumerate(data_):
        print(f'Klar med {i+1}/{len(data_)}')

        identifier, result = y
        if identifier in identifier_set:
            continue

        identifier_set.add(identifier)
        samples = identifier[6]

        x = result['iteration_params_all']
        fun = result['expectation_vals_all']
        eig = result['correct']

        fun_evals = np.round(max_meas/samples)
        fun_evals[fun_evals==0] = 1

        fel_none = NM_fel_measmax.fel_measmax(x, fun, identifier, fun_evals)

        error = (eig - fel_none) / eig * 100

        if samples not in samples_lst:
            samples_lst.append(samples)
            print(samples_lst)
            fel.append(error)
            nr.append(1)
        elif samples in samples_lst:
            idx = np.argmin(np.abs(np.array(samples_lst) - samples))
            fel[idx] = (fel[idx] + error)
            nr[idx] +=1
        print(nr)

    fel = np.array(fel)
    print(nr)
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

    data_ = data.load(data_file, base_dir)[0]

    fel = np.zeros([36, 60])
    samples_lst = []
    max_meas_lst = []
    nr = np.zeros([36, 60])

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

    for j in range(36):
        for k in range(60):
            if np.any(nr[j,k] != 0):
                fel[j,k]/=nr[j,k]
            else:
                fel[j,k] = 5

        print(fel.shape[0])
    file = f'NM_heatmap/v{version}/{ansatz_name}_{minimizer}_size={size}.pkl'
    data2 = np.linspace(1e6, 3e6, 41), np.linspace(1e4, 3e5, 41), fel
    data.save(file, data2, extract=True)


version = 4
size = 3
ansatz_name = 'multi_particle'
minimizer = 'nelder-mead'

NM_save(version, size, ansatz_name, minimizer)