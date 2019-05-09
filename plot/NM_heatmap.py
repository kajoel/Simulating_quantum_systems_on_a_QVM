# @author = Carl, 30/4
# Imports
import numpy as np
import matplotlib.pyplot as plt

from core import data
from plot import tikzfigure

version = 4
heatmap_version = 2
size = 3
ansatz_name = 'multi_particle'
minimizer = 'bayes'

file = f'NM_heatmap/v{heatmap_version}/{ansatz_name}_{minimizer}_size={size}.pkl'
data_ = data.load(file)[0]
max_meas, samples, fel = data_


def print_coord():
    for i, sample in enumerate(samples):
        for j, max_meas_ in enumerate(max_meas):
            if sample<=256500:
                if 4 < int(round(max_meas_/sample))<=300:
                    print(f'{max_meas_}\t{sample}\t{np.log10(abs(fel[i,j]))}')
                else:
                    print(f'{max_meas_}\t{sample}\tnan')
        print('\n')


def plot_heatmap():
    plt.figure(1)
    plt.pcolormesh(max_meas, samples, fel, cmap='viridis', vmin=0,  vmax=1)
    plt.title(ansatz_name.replace('_', ' '))
    plt.colorbar()
    plt.show()

print_coord()
plot_heatmap()



