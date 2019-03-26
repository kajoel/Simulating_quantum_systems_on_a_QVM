"""
@author Eric
This script generates data for a None AND samples run fo a given j, V, and
samples. The inputs are (with default values)
Quasi-spin quantum number j (2)
Index of upper/lower block in j-matrix (0)
Value of potential V (1)
Number of samples on the Quantum Computer (1000)
Inputs should be spaced, e.g.;
pyhton -m meas.none_min_sample 2 0 1 1000
Don't forget to run conda env first
"""

import sys
from core import lipkin_quasi_spin
from grove.pyvqe.vqe import VQE
import numpy as np
from pyquil import get_qc
from meas.sweep import sweep
from scipy.optimize import minimize
from core import ansatz
from core import matrix_to_op
from core import vqe_eig
from core import init_params
import matplotlib.pyplot as plt

if len(sys.argv) is not 5:
    j = 2
    i = 0
    V = 1
    samples = 1000
else:
    j = float(sys.argv[1])
    i = int(sys.argv[2])
    V = float(sys.argv[3])
    samples = int(sys.argv[4])

print('Values set to:', '\nj: ', j, '\ni: ', i, '\nV: ', V, '\nsamples: ',
      samples)

h = lipkin_quasi_spin.hamiltonian(j, V)[i]
no_of_qubits = int.bit_length(h.shape[0])
qc = get_qc('{}q-qvm'.format(no_of_qubits))

sweep_meas_none = sweep(h, qc, ansatz_=ansatz.multi_particle,
                        matrix_operator=matrix_to_op.multi_particle,
                        num_para=10,
                        start=-5, stop=5, samples=None)
sweep_meas = sweep(h, qc, ansatz_=ansatz.multi_particle,
                   matrix_operator=matrix_to_op.multi_particle,
                   num_para=10,
                   start=-5, stop=5, samples=samples)

vqe = VQE(minimizer=minimize, minimizer_kwargs={'method': 'Nelder-Mead'})

if samples < 10000:
    fatol = 1e-1
else:
    fatol = 5e-2
print('fatol set to: ', fatol)
H = matrix_to_op.multi_particle(h)
min_eig, opt_param = vqe_eig.smallest(H, qc=qc,
                                      initial_params=init_params.alternate(
                                          h.shape[0]), disp_run_info=True,
                                      fatol=fatol, num_samples=samples)
min_eig_exp = vqe.expectation(ansatz.multi_particle(opt_param), H,
                              samples=samples,
                              qc=qc)
print('min_eig_exp:', min_eig_exp)