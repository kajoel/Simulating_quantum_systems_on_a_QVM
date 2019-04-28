

from core import ansatz, matrix_to_op, vqe_override, vqe_eig, lipkin_quasi_spin, \
    init_params
from core import callback as cb
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import numpy as np
import csv
from meas import liveplot

samples = 10000
j = 5 / 2
V = 1
i = 1
h = lipkin_quasi_spin.hamiltonian(j, V)[i]
print(h.toarray())
eigvals = lipkin_quasi_spin.eigs(j, V)[i]
print(eigvals)
qc = ansatz.multi_particle_qc(h)

disp_options = {'disp': True, 'xatol': 1e-3, 'fatol': 1e-4,
                'maxiter': 1000}
vqe = vqe_override.VQE_override(minimizer=minimize,
                                minimizer_kwargs={'method': 'Nelder-Mead',
                                                  'options': disp_options})

# Generate background
start = -5
stop = 5
with open('sweepsurface.csv', 'r') as readFile:
    reader = csv.reader(readFile)
    expval = list(reader)

readFile.close()
expval = np.asarray([[float(x) for x in y] for y in expval])
params = np.linspace(start, stop, len(expval))
X, Y = np.meshgrid(params, params)

# Do the liveplot
plot = liveplot.Liveplot()
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
color = (0, 204 / 255, 204 / 255)
surf = plot.ax.plot_surface(X, Y, expval, color=color, linewidth=0, alpha=0.5)


energies = vqe_eig.smallest(matrix_to_op.multi_particle(h), qc,
                            init_params.zeros(
                                h.shape[0]), vqe,
                            ansatz_=ansatz.multi_particle(h),
                            samples=None,
                            callback=cb.merge_callbacks(plot.scatter3d(),
                                                        cb.stop_if_stuck_x_times(
                                                            5)))

plt.show()
'''
for angle in range(0, 360, 1):
        plot.ax.view_init(elev=10., azim=angle)
        plot.figure.draw()
        #plot.figure.pause(0.05)
'''

