# @author Sebastian

from core import lipkin_quasi_spin, init_params
from pyquil import get_qc

from core import vqe_eig
from core import ansatz
from core import callback as cb

import numpy as np
from core import matrix_to_op

samples = 5000
sweep_params = 30

qc = get_qc('3q-qvm')
j, V = 1, 1
h = lipkin_quasi_spin.hamiltonian(j, V)[0]


# plot = liveplot.Liveplot()
# vqe_analysis.sweep_parameters(h, qc, new_version=True, samples=None,
#                               num_para=sweep_params, start=-3, stop=3, callback=
#                               plot.plotline, plot=False)


energies = vqe_eig.smallest(matrix_to_op.multi_particle(h), qc,
                            init_params.ones(
                                h.shape[0]),
                            ansatz_=ansatz.multi_particle(h.shape[0]),
                            samples=samples,
                            fatol=1e-1 * 16 / np.sqrt(samples),
                            return_all_data=True,
                            callback=cb.merge_callbacks(cb.scatter(),
                                                        cb.stop_if_stuck_x_times(2)))
print(energies)