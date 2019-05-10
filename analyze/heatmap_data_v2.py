# @author = Carl, 9/5
# Imports
import numpy as np
from constants import ROOT_DIR
from os.path import join
from core import data
from analyze import NM_fel_measmax


def bayes_save(version, size, ansatz_name, matidx=None):
    base_dir = join(ROOT_DIR, f'data/final_bayes/v{version}')
    data_file1 = f'{ansatz_name}_bayes_size={size}_part_1.pkl'
    data_file2 = f'{ansatz_name}_bayes_size={size}_part_2.pkl'

    data1 = data.load(data_file1, base_dir)[0]
    data2 = data.load(data_file2, base_dir)[0]

    data_ = data1 + data2

    fel = np.zeros([36, 60])
    nr = np.zeros([36, 60])

    samples_lst = np.zeros([36])
    max_meas_lst = np.zeros([60])
    fun_evals_lst = np.zeros([36, 60])

    identifier_set = set()

    for i, y in enumerate(data_):
        identifier, result = y

        samples = identifier[-1]
        max_meas = identifier[-2]
        fun_evals = int(round(max_meas / samples))

        if identifier in identifier_set or samples > 256500 or not (
                matidx != None and matidx == identifier[4]):
            continue


        identifier_set.add(identifier)

        samples_idx = np.argmin(np.abs(np.linspace(2750, 256500, 36) - samples))
        max_meas_idx = np.argmin(np.abs(np.linspace(50000, 3e6, 60) - max_meas))

        eig = result['correct']
        fun = result['fun']

        error = abs((eig - fun) / eig * 100)

        max_meas_lst[max_meas_idx] = max_meas
        samples_lst[samples_idx] = samples
        fun_evals_lst[samples_idx, max_meas_idx] = fun_evals

        if 4 < int(round(max_meas / samples)) <= 300:
            fel[samples_idx, max_meas_idx] += error
            nr[samples_idx, max_meas_idx] += 1

    for i in range(36):
        for j in range(60):
            if nr[i, j] != 0:
                fel[i, j] /= nr[i, j]

    for i in range(36):
        idx_dict = {}
        for idx, fun_evals in enumerate(fun_evals_lst[i, :]):
            if fun_evals not in idx_dict:
                idx_dict[fun_evals] = []
            idx_dict[fun_evals].append(idx)

        for fun_evals, idx_lst in zip(idx_dict.keys(), idx_dict.values()):
            fel[i, idx_lst] = np.mean(fel[i, idx_lst])

    for row in nr:
        print(row)

    if matidx == None:
        file = f'heatmap_data/v2/bayes_{ansatz_name}_size={size}.pkl'
    else:
        file = f'heatmap_data/v2/bayes_{ansatz_name}_size={size}_matidx={matidx}.pkl'
    data2 = np.linspace(50000, 3e6, 60), np.linspace(2750, 256500, 36), fel
    data.save(file, data2, extract=True)


def NM_save(version, size, ansatz_name, matidx=None):
    base_dir = join(ROOT_DIR, f'data/final_nm/v{version}')
    data_file = f'{ansatz_name}_nelder-mead_size={size}.pkl'

    data_ = data.load(data_file, base_dir)[0]

    max_meas_lst = np.linspace(50000, 3e6, 60)
    samples_lst = np.linspace(2750, 256500, 36)
    fel = np.zeros([36, 60])
    nr = np.zeros([36, 60])

    identifier_set = set()

    for y in data_:
        identifier, result = y
        samples = identifier[-2]
        if identifier in identifier_set or samples > 256500 or not (
                matidx != None and matidx == identifier[4]):
            continue

        identifier_set.add(identifier)

        i = np.argmin(np.abs(samples_lst - samples))

        for j, max_meas in enumerate(max_meas_lst):
            fun_evals = int(round(max_meas / samples))
            if fun_evals <= 4:
                continue

            funs = np.array(result['expectation_vals_all'][:fun_evals])
            params = np.array(result['iteration_params_all'][:fun_evals])

            idx = np.argmin(funs)
            x = params[idx, :]

            idx_lst = np.linalg.norm(params - x, axis=1) <= 1e-3

            fun = np.mean(funs[idx_lst])
            eig = result['correct']

            fel[i, j] += np.abs((fun - eig) / eig * 100)
            nr[i, j] += 1

    for rows in nr:
        print(rows)

    for i in range(36):
        for j in range(60):
            if nr[i, j] != 0:
                fel[i, j] /= nr[i, j]

    if matidx == None:
        file = f'heatmap_data/v2/nm_{ansatz_name}_size={size}.pkl'
    else:
        file = f'heatmap_data/v2/nm_{ansatz_name}_size={size}_matidx={matidx}.pkl'
    data2 = max_meas_lst, samples_lst, fel
    data.save(file, data2, extract=True)


version = 4
size = 3
ansatz_name = 'multi_particle'

NM_save(version, size, ansatz_name, 0)
