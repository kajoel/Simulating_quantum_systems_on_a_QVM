from core.lipkin_quasi_spin import hamiltonian, eigs
from grove.pyvqe.vqe import VQE
import numpy as np
from pyquil import get_qc
from scipy.optimize import minimize
from core import ansatz
from core import matrix_to_op
from core import vqe_eig
from core import init_params
import matplotlib.pyplot as plt
from core import vqeOverride
from core.vqe_eig import smallest
from old.mutable_num import MutableNum
from matplotlib import cm
from pyquil.api import WavefunctionSimulator
from mpl_toolkits.mplot3d import Axes3D
from core.init_params import ones

j = 1
V = 1
i = 0
h = hamiltonian(j, V)[i]
print(h.toarray())
eigvals = eigs(j, V)[i]
print(eigvals)
qc = get_qc('3q-qvm')
H = matrix_to_op.multi_particle(h)

samples = MutableNum(2000)
count = MutableNum(0)


def callback(*args, **kwargs):
    print(*args)
    count.value += 1
    if count < 80:
        samples.value += 500
    plt.errorbar(args[0], args[1], 3*np.sqrt(args[2]), capthick=True)
    plt.pause(0.05)


res = smallest(H, qc, ones(h.shape[0]),
               ansatz_=ansatz.multi_particle(h.shape[0]),
               samples=samples,
               opt_algorithm='Nelder-Mead',
               maxiter=10000,
               disp=callback,
               display_after_run=True,
               xatol=1e-2, fatol=1e-1,
               return_all_data=True,
               convert_op=matrix_to_op.multi_particle,
               print_option=None)

params = np.array(res['iteration_params'])
exp_vals = np.array(res['expectation_vals'])
exp_std = np.sqrt(np.array(res['expectation_vars']))

#plt.errorbar(x=params, y=exp_vals, yerr=exp_std, fmt='.')
