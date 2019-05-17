from core import ansatz, matrix_to_op, vqe_override, vqe_eig, lipkin_quasi_spin, \
    init_params, interface
from core import callback as cb
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import numpy as np
import csv
from plot import liveplot

import plotly.plotly as py
import plotly.graph_objs as go

import numpy as np
import pandas as pd
import plotly

z_data = pd.read_csv('Sweep_surface_j2.5i0.csv')
x_data = np.linspace(-5, 5, 200)
y_data = np.linspace(-5, 5, 200)
z_data = np.asarray(z_data)
z_data = z_data[:, 1:]
print(z_data.shape)

z_offset = (np.min(z_data) - 2) * np.ones(z_data.shape)
proj_z = lambda x_data, y_data, z_data: z_data  # projection in the z-direction
colorsurfz = proj_z(x_data, y_data, z_data)

data1 = go.Surface(
    x=x_data,
    y=y_data,
    z=z_data,
    opacity=0.8,
    showscale=False,
    colorscale='Viridis',
    cauto=False,
    cmin=-6,
    cmax=0
)

proj = go.Surface(z=list(z_offset),
                  x=list(x_data),
                  y=list(y_data),
                  showscale=False,
                  surfacecolor=colorsurfz,
                  colorscale='Viridis',
                  cauto=False,
                  cmin=-6,
                  cmax=0
                  )

samples = 10000
j = 5 / 2
V = 1
i = 0
h = lipkin_quasi_spin.hamiltonian(j, V)[i]
print(h.toarray())
eigvals = lipkin_quasi_spin.eigs(j, V)[i]
print(eigvals)

disp_options = {'disp': True, 'xatol': 1e-3, 'fatol': 1e-4,
                'maxiter': 1000}
vqe = vqe_override.VQE_override(minimizer=minimize,
                                minimizer_kwargs={'method': 'Nelder-Mead',
                                                  'options': disp_options})
H, qc, ansatz_, initial_params = interface.create_and_convert('multi_particle',
                                                              h)

bad_initial_params = np.array([-1, -1])
energies = vqe_eig.smallest(matrix_to_op.multi_particle(h), qc,
                            bad_initial_params, vqe,
                            ansatz_=ansatz.multi_particle_stereographic(h),
                            samples=None,
                            callback=cb.stop_if_stuck_x_times(5))

dots_params = energies['iteration_params']
dots_vals = energies['expectation_vals']

# Make fewer points
'''
reduction = int(len(dots_params) / 5)
dots_params = [dots_params[5 * i] for i in range(reduction)]
dots_vals = [dots_vals[5 * i] for i in range(reduction)]
'''
xvec = [dots_params[i][0] for i in range(len(dots_params))]
yvec = [dots_params[i][1] for i in range(len(dots_params))]
zvec = [dots_vals[i] for i in range(len(dots_vals))]

xvec = np.asarray(xvec)
yvec = np.asarray(yvec)
zvec = np.asarray(zvec)
# print(xvec)
# print(yvec)
# print(zvec)


data2 = go.Scatter3d(
    x=yvec,
    y=xvec,
    z=zvec,
    # mode='markers',
    showlegend=False,
    marker=dict(
        # color='rgb(127, 127, 127)',
        color='red',
        size=6,
        symbol='circle',
        line=dict(
            # color='rgb(204, 204, 204)',
            color='black',
            width=2
        ),
        opacity=0.9
    )
)

zbase = (np.min(zvec) - 2) * np.ones(zvec.shape)
proj2 = go.Scatter3d(
    x=yvec,
    y=xvec,
    z=zbase,
    showlegend=False,
    # mode='markers',
    marker=dict(
        # color='rgb(127, 127, 127)',
        color='red',
        size=2,
        symbol='circle',
        line=dict(
            # color='rgb(204, 204, 204)',
            color='black',
            width=2
        ),
        opacity=0.9
    )
)
data = [data1, data2, proj, proj2]
layout = go.Layout(
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0,
    ),
    scene=dict(
        xaxis=dict(
            nticks=5, range=[-5, 0], title=r'$\\theta_1$'),
        yaxis=dict(
            nticks=5, range=[-2, 3], title=r'$\\theta_2$'),
        zaxis=dict(
            nticks=8, range=[-8, 7], title='Energi')
    )
)
figure = go.Figure(data=data, layout=layout)
py.iplot(figure, filename='j2.5i0_NM_titles', include_mathjax='cdn')
plotly.offline.plot(figure)
'''
# Generate background
start = -5
stop = 5
with open('Sweep_surface_j3i1.csv', 'r') as readFile:
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

bad_initial_params = np.array([-1, -1])
energies = vqe_eig.smallest(matrix_to_op.multi_particle(h), qc,
                            bad_initial_params, vqe,
                            ansatz_=ansatz.multi_particle_stereographic(h),
                            samples=None,
                            callback=cb.merge_callbacks(plot.scatter3d(),
                                                        cb.stop_if_stuck_x_times(
                                                            5)))



plt.show()
'''
'''
for angle in range(0, 360, 1):
        plot.ax.view_init(elev=10., azim=angle)
        plot.figure.draw()
        #plot.figure.pause(0.05)
'''
