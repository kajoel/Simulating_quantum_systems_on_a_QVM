import matplotlib.pyplot as plt
from core import lipkin_quasi_spin, init_params
from pyquil import get_qc

from core import vqe_eig
from core import ansatz
from old import vqe_analysis

import numpy as np
from core import matrix_to_op

samples = 5000
sweep_params = 30

qc = get_qc('3q-qvm')
j, V = 1, 1
h = lipkin_quasi_spin.hamiltonian(j, V)[0]

oldx = None
oldy = None


def testprint2(x, y):
    global oldx
    global oldy
    if oldx is not None and oldy is not None: plt.plot([oldx, x], [oldy, y],
                                                       color='red')
    oldx = x
    oldy = y
    plt.pause(0.05)


figure = plt.figure()
line = plt.plot([], [])[0]


def testprint3(x, y):
    global figure
    global line
    (oldx, oldy) = line.get_data()
    allx = np.concatenate((oldx, x))
    ally = np.concatenate((oldy, [y]))
    line.set_data(allx, ally)
    figure.axes[0].relim()
    figure.axes[0].autoscale_view()
    plt.pause(0.05)


vqe_analysis.sweep_parameters(h, qc, new_version=True, samples=samples,
                              num_para=sweep_params, start=-3, stop=3, callback=
                              testprint3, plot=False)


def testprint(x, y):
    plt.scatter(x, y, color='blue')
    plt.pause(0.05)
    print("Parameter: {}".format(x[0]))
    print("Expectation: {}".format(y))


energies = vqe_eig.smallest(matrix_to_op.multi_particle(h), qc,
                            init_params.ones(
                                h.shape[0]),
                            ansatz_=ansatz.multi_particle(h.shape[0]), samples=samples,
                            disp_run_info=testprint,
                            fatol=1e-1 * 16 / np.sqrt(samples))[0]
plt.show()
