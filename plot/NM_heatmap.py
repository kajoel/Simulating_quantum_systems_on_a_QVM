# @author = Carl, 30/4
# Imports
import numpy as np
import matplotlib.pyplot as plt

from core import data
#from plot import tikzfigure

version = 7
heatmap_version = 2
size = 2
ansatz_name = 'multi_particle'
minimizer = 'nm'

file = f'heatmap_data/v{heatmap_version}/{minimizer}_{ansatz_name}_size={size}.pkl'
data_ = data.load(file)[0]
max_meas, samples, fel = data_


def print_coord():
    import csv
    vec1, vec2, vec3 = [],[],[]
    

    for i, sample in enumerate(samples):
        for j, max_meas_ in enumerate(max_meas):
            vec1.append(max_meas_)
            vec2.append(sample)
            if sample<=256500:
                if 4 < int(round(max_meas_/sample)):
                    vec3.append(np.log10((fel[i,j])))
                    print(f'{max_meas_}\t{sample}\t{np.log10((fel[i,j]))}')
                else:
                    print(f'{max_meas_}\t{sample}\tnan')
                    vec3.append('nan')

        print('\n')
    # Prints the data to a csv file you can copy from
    with open('test.csv', 'w') as f:
        writer = csv.writer(f, delimiter=' ')    
        writer.writerows(zip(vec1, vec2, vec3))


def plot_heatmap():
    plt.figure(1)
    plt.pcolormesh(max_meas, samples, fel, cmap='viridis', vmin=0,  vmax=1)
    plt.title(ansatz_name.replace('_', ' '))
    plt.colorbar()
    plt.show()

print_coord()
plot_heatmap()



