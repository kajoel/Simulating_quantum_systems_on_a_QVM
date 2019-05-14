# @author = Carl, 30/4
# Imports
import numpy as np
import matplotlib.pyplot as plt

from core import data
from plot import tikzfigure

version = 4
heatmap_version = 2
size = 5
ansatz_name = 'multi_particle'
minimizer = 'nm'

file = f'heatmap_data/v{heatmap_version}/{minimizer}_{ansatz_name}_size={size}.pkl'
data_ = data.load(file)[0]
max_meas, samples, fel = data_


def print_coord():
    for i, sample in enumerate(samples):
        for j, max_meas_ in enumerate(max_meas):
            if sample<=256500:
                if 4 < int(round(max_meas_/sample)):
                    print(f'{max_meas_}\t{sample}\t{(fel[i,j])}')
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



