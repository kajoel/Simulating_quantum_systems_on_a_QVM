# @author = Carl, 9/5
# Imports
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

from os.path import join
from constants import ROOT_DIR

from core import data
from plot import tikzfigure

version = 2
size = 4
minimizer = 'nm'
file = f'hero_run_{minimizer}/v{version}/size={size}'

data_, _ = data.load(file)

V_lst = [[], []]
fun_lst = [[], []]

for identifier, result in data_:
    size, hamiltonian_idx, V, max_meas, samples = identifier
    V_lst[hamiltonian_idx-1].append(V)
    fun_lst[hamiltonian_idx-1].append(result['fun'])

V_lst = np.array(V_lst)
fun_lst = np.array(fun_lst)

idx = np.argsort(V_lst, axis=1)
V_lst = np.take_along_axis(V_lst, idx, axis=1)
fun_lst = np.take_along_axis(fun_lst, idx, axis=1)

plt.plot(V_lst[0,:], fun_lst[0,:], '-o')
plt.plot(V_lst[1,:], fun_lst[1,:], '-o')
plt.show()
