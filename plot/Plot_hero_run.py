# @author = Carl, 9/5
# Imports
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

from os.path import join
from constants import ROOT_DIR

from core.interface import hamiltonians_of_size
from core import data
from plot import tikzfigure

version = 2
minimizer = 'nm'

for size in [3,4]:

    file = f'hero_run_{minimizer}/v{version}/size={size}'
    data_, _ = data.load(file)

    V_lst = [[], []]
    fun_lst = [[], []]
    var_lst = [[], []]
    eigs1 = []
    eigs2 = []

    for identifier, result in data_:
        size, hamiltonian_idx, V, max_meas, samples = identifier

        funs = np.array(result['expectation_vals_all'])
        params = np.array(result['iteration_params_all'])
        vars = np.array(result['expectation_vars_all'])

        idx = np.argmin(funs)
        x = params[idx, :]

        idx_lst = np.linalg.norm(params - x, axis=1) <= 1e-3
        fun = np.mean(funs[idx_lst])
        var = np.mean(vars[idx_lst])

        V_lst[hamiltonian_idx-1].append(V)
        fun_lst[hamiltonian_idx-1].append(fun)
        var_lst[hamiltonian_idx - 1].append(var)

    V_lst = np.array(V_lst)
    fun_lst = np.array(fun_lst)
    var_lst = np.array(var_lst)

    idx = np.argsort(V_lst, axis=1)
    V_lst = np.take_along_axis(V_lst, idx, axis=1)
    fun_lst = np.take_along_axis(fun_lst, idx, axis=1)
    var_lst = np.take_along_axis(var_lst, idx, axis=1)

    var_lst = 20*np.sqrt(var_lst)

    for V in V_lst[0,:]:
        h = list(hamiltonians_of_size(size, V)[0][1:3])
        eig1 = np.linalg.eigvalsh(h[0].todense())
        eig2 = np.linalg.eigvalsh(h[1].todense())
        eigs1.append(eig1[eig1 < 0])
        eigs2.append(eig2[eig2 < 0])

    eigs1 = np.array(eigs1)
    eigs2 = np.array(eigs2)

    print(var_lst)

    plt.plot(V_lst[0,:], fun_lst[0,:], '-o', label=f'Size={size} Matidx=1')
    plt.plot(V_lst[1,:], fun_lst[1,:], '-o', label=f'Size={size} Matidx=2')
    plt.plot(V_lst[1,:], eigs1, ':', color='gray')
    plt.plot(V_lst[1,:], eigs2, ':', color='gray')

plt.legend()
tikzfigure.save('hero_run_nm')
plt.show()
