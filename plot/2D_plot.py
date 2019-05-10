# @author = Carl, 9/5
# Imports
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

from os.path import join
from constants import ROOT_DIR

from core import data
from plot import tikzfigure

version = 4
heatmap_version = 2
size = 3
ansatz_name = 'multi_particle'
minimizer = 'bayes'
path = f'heatmap_data/v{heatmap_version}'

matidx=None

if matidx == None:
    file = f'/{minimizer}_{ansatz_name}_size={size}.pkl'

else:
    file = f'/{minimizer}_{ansatz_name}_size={size}_matidx={matidx}.pkl'

file = path + file
data_ = data.load(file)[0]
max_meas, samples, fel = data_

fel [fel==0] = 10000

samples_lst = []
fel_lst = []

for j, max_meas_ in enumerate(max_meas):
    idx = np.argmin(fel[:,j])
    samples_lst.append(samples[idx])
    fel_lst.append(fel[idx,j])

#fit = np.polyfit(max_meas + np.log(max_meas), samples_lst, 1)

def func(x, a, b):
    return a*x + b

print(samples_lst)

#c, _ = curve_fit(func, max_meas, samples_lst)
c = np.polyfit(max_meas, samples_lst, 1)
samples_fit = func(max_meas, c[0], c[1])

print(samples_fit)

plt.figure(1)
plt.plot(max_meas, samples_lst, 'o', label=f'{matidx}')
plt.plot(max_meas, samples_fit)

fel_lst2 = []
for j, max_meas_ in enumerate(max_meas):
    idx = np.argmin(np.abs(samples - samples_fit[j]))
    fel_lst2.append(np.log10(fel[idx, j]))

'''
for j in range(60):
    print(f'{max_meas[j]}\t{samples_fit[j]}')
'''

plt.figure(2)
plt.plot(max_meas, fel_lst2)

plt.show()


print(func(0.5e6, c[0], c[1]))

'''
def print_coord():
    for i, sample in enumerate(samples):
        for j, max_meas_ in enumerate(max_meas):
            if sample<=256500:
                if 4 < int(round(max_meas_/sample)):
                    print(f'{max_meas_}\t{sample}\t{np.log10((fel[i,j]))}')
                else:
                    print(f'{max_meas_}\t{sample}\tnan')
        print('\n')



print_coord()
'''