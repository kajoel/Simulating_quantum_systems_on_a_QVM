# @author = Carl, 30/4
# Imports
import numpy as np
import matplotlib.pyplot as plt

from core import data
from plot import tikzfigure

version = 3
size = 3
ansatz_name = 'one_particle_ucc'
minimizer = 'bayes'

file = f'NM_heatmap/v{version}/{ansatz_name}_{minimizer}_size={size}.pkl'
data_ = data.load(file)[0]
max_meas, samples, fel = data_


def print_coord():
    for i, sample in enumerate(samples):
        for j, max_meas_ in enumerate(reversed(max_meas)):
            if sample<=256500:
                if int(round(max_meas_/sample))>4:
                    print(f'{max_meas_}\t{sample}\t{fel[i,-j]}')
                else:
                    print(f'{max_meas_}\t{sample}\trgb=1,1,1')
        print('\n')


def plot_heatmap():
    plt.figure(1)
    plt.pcolormesh(max_meas, samples, fel, cmap='viridis', vmin=z_min,  vmax=5)
    plt.title(ansatz_name.replace('_', ' '))
    plt.colorbar()
    plt.show()



