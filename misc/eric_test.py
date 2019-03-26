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
from matplotlib import cm
from pyquil.api import WavefunctionSimulator
from mpl_toolkits.mplot3d import Axes3D

samples = 10000
j = 2
V = 1
h = hamiltonian(j, V)[1]
print(h.toarray())
eigvals = eigs(j, V)[1]
print(eigvals)
qc = get_qc('3q-qvm')
H = matrix_to_op.multi_particle(h)
vqe = VQE(minimizer=minimize, minimizer_kwargs={'method': 'Nelder-Mead'})
sweep_steps = 50
parameters = np.linspace(-5, 5, sweep_steps)

exp_val = [
    vqe.expectation(ansatz.multi_particle(np.array([para])), H, samples=None,
                    qc=qc) for para in parameters]
'''
exp_val = np.zeros((sweep_steps, sweep_steps))
mesh_1 = np.zeros((sweep_steps, sweep_steps))
mesh_2 = np.zeros((sweep_steps, sweep_steps))

for i, p_1 in enumerate(parameters):
    mesh_1[i] += p_1
    mesh_2[i] = parameters
    exp_val[i] = [
        vqe.expectation(ansatz.multi_particle(np.array([p_1, p_2])), H,
                        samples=None, qc=qc)
        for p_2 in parameters]
    fig = plt.figure(0)
    ax = fig.add_subplot(111, projection='3d')
    # Plot the surface

    # save_run_to_csv(exp_val)
    ax.plot_surface(mesh_1, mesh_2, exp_val, cmap=cm.coolwarm)

# min_eig = vqe.vqe_run(one_particle, H,
#  initial_params=init_params.alternate(h.shape[0]),
#  samples=samples, qc=qc, disp=print)
'''

min_eig, optparam = vqe_eig.smallest(H, qc=qc, ansatz_=ansatz.multi_particle,
                                     num_samples=samples,
                                     fatol=1e-2,
                                     initial_params=init_params.ones(
                                         h.shape[0]),
                                     disp_run_info=True)
print('Min eig vqe: ', min_eig)
min_eig_exp = vqe.expectation(ansatz.multi_particle(optparam), H, samples=10000,
                              qc=qc)
print('Min eig vqe_exp: ', min_eig_exp)
# plt.show()

exp_val2 = [
    vqe.expectation(ansatz.multi_particle(np.array([para])), H, samples=samples,
                    qc=qc) for para in parameters]

plt.figure(0)
plt.plot(parameters, exp_val, 'black', lw=2, label='Samples: None')
plt.plot(parameters, exp_val2, 'orange', lw=2, ls='dashed',
         label='Samples: {}'.format(samples))
plt.plot(optparam, min_eig_exp, '*', ms=20,
         label='Min value: {} \n True value: {}'.format(round(min_eig_exp, 4),
                                                        round(min(eigvals), 4)))
plt.xlabel('Parameter value')
plt.ylabel('Expected value of Hamiltonian')
plt.legend(bbox_to_anchor=(0.01, 0.99), loc=2, borderaxespad=0.)
plt.show()
