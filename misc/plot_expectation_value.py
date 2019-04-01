# Carl 12/3
###############################################################################
# Imports
from core.lipkin_quasi_spin import hamiltonian, eigs
from grove.pyvqe.vqe import VQE
import numpy as np
from pyquil import get_qc
from scipy.optimize import minimize
from core import ansatz
from core import matrix_to_op
from core import vqe_eig
from core import init_params
from meas import sweep
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

###############################################################################
samples = 10000
matrix = 0
j = 1
V = 1
h = hamiltonian(j, V)[matrix]
dim = h.shape[0]
print(h.toarray())
print('\n')
eigvals = eigs(j, V)[matrix]
print(eigvals)
print('\n')
qc = get_qc(str(int.bit_length(h.shape[0])) + 'q-qvm')
#qc = get_qc(str(h.shape[0]) + 'q-qvm')
###############################################################################
H = matrix_to_op.multi_particle(h)
initial_params = init_params.ucc(dim)
ansatz_ = ansatz.multi_particle_ucc(dim)
###############################################################################
# Smallest samples=None

result_None = vqe_eig.smallest(H, qc, initial_params, ansatz_=ansatz_,
                               samples=None, fatol=1e-3, xatol=1e-2,
                               disp_run_info=True,
                               display_after_run=True, return_all_data=True)
###############################################################################
# Sweep

sweep = sweep.sweep(h, qc, ansatz_, matrix_to_op.one_particle,
                               start=-10, stop=10)
###############################################################################
# Plot


def plot_(result, sweep):
    # 3D
    exp_vals = result_None['expectation_vals']
    iter_params = result_None['iteration_params']

    tmp = np.zeros([len(iter_params), len(iter_params[0])])
    for i, iter_param in enumerate(iter_params):
        tmp[i] = iter_param

    iter_params = tmp


    if iter_params.shape[1] == 1:
        fig = plt.figure(1)
        x = np.asarray(iter_params)
        y = np.asarray(exp_vals)
        plt.plot(sweep[1],sweep[0])

        for x, y in zip(iter_params, exp_vals):
            plt.pause(0.5)
            plt.scatter(x, y, color='r')

        plt.show()

    elif iter_params.shape[1] == 2:
        fig = plt.figure(1)
        ax = fig.add_subplot(111, projection='3d')

        x = np.asarray(iter_params[0])
        y = np.asarray(iter_params[1])
        z = np.asarray(exp_vals)

        # Plot the surface.
        surf = ax.plot_surface(sweep[1],sweep[2], sweep[0], cmap=cm.coolwarm)
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        # plt.plot(parameters, eigs)

        for x, y in zip(iter_params, exp_vals):
            ax.scatter(x[0], x[1], y, color='r')
            plt.pause(0.5)

        plt.show()

###############################################################################

plot_(result_None,sweep)




'''
result = vqe_eig.smallest(H, qc, initial_params, ansatz_, samples=samples,
                         disp_run_info=True, display_after_run=False,
                          xatol=1e-2, fatol=1e-3)
###############################################################################
parameter = result[1]
ansatz_ = ansatz_(parameter)
vqe = VQE(minimizer=minimize, minimizer_kwargs={'method': 'Nelder-Mead'})

n=20
eig = 0
for i in range(n):
    eig += vqe.expectation(ansatz_, H, samples=samples, qc=qc)

eig = eig/n

print('\n', 'Eigenvalue after mean:',eig)
###############################################################################
'''
