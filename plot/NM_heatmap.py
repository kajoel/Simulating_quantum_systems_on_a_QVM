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

# version = 3
# size = 3
# ansatz_name = 'multi_particle'
# minimizer = 'nelder-mead'
# base_dir = join(ROOT_DIR, f'data/final_nm/v{version}')
# file = f'{ansatz_name}_{minimizer}_size={size}.pkl'
file = 'NM_heatmap/v1.pkl'
data_ = data.load(file)[0]
max_meas, samples, fel = data_

print(fel)

z_min = 0
z_max = 20

y_min = np.min(samples)
y_max = np.max(samples)

x_min = np.min(max_meas)
x_max = np.max(max_meas)

plt.figure(1)
plt.pcolormesh(max_meas, samples, fel, cmap='viridis', vmin=z_min, vmax=z_max)
plt.colorbar()

plt.figure(2)
plt.pcolor(fel)
plt.colorbar()

plt.show()

'''
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
