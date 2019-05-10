# Carl 4/5
###############################################################################
# Imports
from core import data
import numpy as np
import matplotlib.pyplot as plt
from constants import ROOT_DIR
from os.path import join
import itertools
from plot import tikzfigure


def meas_fel(version, size, mat_idx):
    base_dir = join(ROOT_DIR, f'data/NM_Restart_Parallel/v{version}')
    meas, fel, status = {}, {}, {}

    file = f'multi_particle_size={size}_matidx={mat_idx}.pkl'
    data_, _ = data.load(file, base_dir)

    # meas_tmp = {}
    # fel_tmp = {}
    # status_tmp = {}

    for i, x in enumerate(data_):
        identifier, result = x

        samples = identifier[3]
        if version == 5:
            max_para = 0
            repeats = identifier[4]

        else:
            max_para = identifier[4]
            repeats = identifier[5]

        eig = result['correct']
        fun_none = result['fun_none']
        fun_evals = result['fun_evals']

        try:
            meas[max_para]
        except:
            meas[max_para] = []
            fel[max_para] = []
            status[max_para] = []
        if repeats <= 1:
            meas[max_para].append(samples * fun_evals)
            fel[max_para].append(np.abs((fun_none - eig) / eig) * 100)
            status[max_para].append(result['status'])

        # meas[mat_idx], fel[mat_idx], status[mat_idx] = meas_tmp, fel_tmp, status_tmp

    return meas, fel

version=7
size = 4
mat_idx = 0
ansatz_name ='multi_particle'

base_dir = join(ROOT_DIR, f'data/NM_Restart_Parallel/v{version}')

file = f'{ansatz_name}_size={size}_matidx={mat_idx}.pkl'
data_, metadata= data.load(file, base_dir)

error_lst = []
samples_lst = []
fun_lst = []
eig=0
for i, x in enumerate(data_):
    identifier, result = x
    print(identifier)
    repeats = identifier[5]
    #size = identifier[3]
    #mat_idx = identifier[4]
    #max_meas = identifier[5]
    samples = identifier[3]

    if repeats==0:
        eig = result['correct']
        x = result['x']
        exp_vals = result['expectation_vals_all']
        iter_params = result['iteration_params_all']

        idx = [i for i, y in enumerate(iter_params) if np.linalg.norm(y-x)<1e-3]
        print(exp_vals[idx])
        fun = np.mean(exp_vals[idx])

        exp_vars = np.mean(result['expectation_vars_all'][idx])
        error = 2*np.sqrt(exp_vars)

        samples_lst.append(samples)
        error_lst.append(error)
        fun_lst.append(fun)

sort_idx = np.argsort(samples_lst)

samples_lst = np.array(samples_lst)[sort_idx]
error_lst = np.array(error_lst)[sort_idx]
fun_lst = np.array(fun_lst)[sort_idx]

print(samples_lst)

plt.hlines(y=eig,xmin=0, xmax=samples_lst[-1]*1.05, linewidth=1,
           color='r', linestyles='--')
plt.errorbar(samples_lst, fun_lst, error_lst, marker='o', markersize=3,
             linestyle='None', solid_capstyle='projecting', capsize=2)
plt.legend(['Egenvärde', 'Data'])

for i, _ in enumerate(samples_lst):
    if samples_lst[i]:
        print(f'{samples_lst[i]}\t{fun_lst[i]}\t{error_lst[i]}')


print(eig)
plt.show()