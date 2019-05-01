# @author = Carl, 30/4
# Imports
import numpy as np
import matplotlib.pyplot as plt
import itertools
from constants import ROOT_DIR
from os.path import join

from core import data
from analyze import NM_fel_measmax
from plot import tikzfigure

version = 3
size = 3
ansatz_name = 'multi_particle'
minimizer = 'nelder-mead'

file = f'NM_heatmap/v{version}/{ansatz_name}_{minimizer}_size={size}.pkl'
data_ = data.load(file)[0]
max_meas, samples, fel = data_

z_min = 0
z_max = 20

y_min = np.min(samples)
y_max = np.max(samples)

x_min = np.min(max_meas)
x_max = np.max(max_meas)

plt.figure(1)
plt.pcolormesh(max_meas, samples, fel, cmap='viridis', vmin=z_min, vmax=30)
plt.title(ansatz_name.replace('_', ' '))
plt.colorbar()

version = 3
size = 3
ansatz_name = 'one_particle_ucc'
minimizer = 'nelder-mead'

file = f'NM_heatmap/v{version}/{ansatz_name}_{minimizer}_size={size}.pkl'
data_ = data.load(file)[0]
max_meas, samples, fel = data_

np.savetxt('fel_nelder-mead_ucc.csv', fel, delimiter=",")
np.savetxt('samples_nelder-mead_ucc.csv', samples, delimiter=",")
np.savetxt('max_meas_nelder-mead_ucc.csv', max_meas, delimiter=",")

z_min = 0
z_max = 20

y_min = np.min(samples)
y_max = np.max(samples)

x_min = np.min(max_meas)
x_max = np.max(max_meas)
'''
plt.figure(2)
plt.pcolormesh(max_meas, samples, fel, cmap='viridis', vmin=z_min, vmax=30)
plt.colorbar()
plt.title(ansatz_name.replace('_', ' '))
'''
tikzfigure.save('heatmap_test')
plt.show()

'''
plt.figure(2)
plt.pcolor(fel)
plt.colorbar()
ax = axs[1, 0]
c = ax.imshow(z, cmap='RdBu', vmin=z_min, vmax=z_max,
              extent=[x.min(), x.max(), y.min(), y.max()],
              interpolation='nearest', origin='lower')
ax.set_title('image (nearest)')
fig.colorbar(c, ax=ax)

ax = axs[0, 1]
c = ax.pcolormesh(x, y, z, cmap='RdBu', vmin=z_min, vmax=z_max)
ax.set_title('pcolormesh')
# set the limits of the plot to the limits of the data
ax.axis([x.min(), x.max(), y.min(), y.max()])
fig.colorbar(c, ax=ax)

'''
