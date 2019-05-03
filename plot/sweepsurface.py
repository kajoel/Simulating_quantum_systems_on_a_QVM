from core.lipkin_quasi_spin import hamiltonian, eigs
from core import vqe_override
from pyquil import get_qc
from old import sweep
from core import ansatz
from core import matrix_to_op
from core import interface
from scipy.optimize import minimize
import seaborn
from pandas import DataFrame
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np
import csv

num_para = 200
samples = None
j = 5 / 2
V = 1
i = 0

h = hamiltonian(j, V)[i]
print(h.toarray())
eigvals = eigs(j, V)[i]
print(eigvals)
H, qc, ansatz_, initial_params = interface.create_and_convert('multi_particle',
                                                              h)
vqe = vqe_override.VQE_override(minimizer=minimize,
                                minimizer_kwargs={'method': 'Nelder-Mead'})

exp_val = np.zeros((num_para, num_para))
mesh_1 = np.zeros((num_para, num_para))
mesh_2 = np.zeros((num_para, num_para))
parameters = np.linspace(-5, 5, num_para)

xparams = np.linspace(-5, 0, num_para)
yparams = np.linspace(-2, 3, num_para)

for i in range(num_para):
    # mesh_1[i] += p_1
    # mesh_2[i] = parameters

    exp_val[i] = [
        vqe.expectation(ansatz_(np.array([xparams[i], p_2])), H, samples=None,
                        qc=qc)[
            0]
        for p_2 in yparams]
    print('Done with sweep number {}/{}'.format(i + 1, len(parameters)))

# print(exp_val)

# parameters = [np.ndarray.tolist(parameters)]
# exp_val = np.ndarray.tolist(exp_val)
# newparams = [[None]+parameters[0]]
# newdat = [[parameters[0][i]] + exp_val[i] for i in range(len(parameters[0]))]

with open('Sweep_surface_j2.5i0_newlimits.csv', 'w') as writeFile:
    writer = csv.writer(writeFile)
    # writer.writerows(newparams)
    writer.writerows(exp_val)

writeFile.close()
