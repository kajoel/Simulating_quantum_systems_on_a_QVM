# @author = Carl, 30/4
# Imports
import numpy as np
import matplotlib.pyplot as plt

from core import data
from plot import tikzfigure

version = 3
size = 3
ansatz_name = 'one_particle_ucc'
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


for j, max_meas_ in enumerate(max_meas):
    for i, sample in enumerate(samples):
        print(f'{max_meas_}\t{sample}\t{fel[i][j]}')
    print('\n')


plt.figure(1)
plt.pcolormesh(max_meas, samples, fel, cmap='viridis', vmin=z_min, vmax=30)
plt.title(ansatz_name.replace('_', ' '))
plt.colorbar()
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
