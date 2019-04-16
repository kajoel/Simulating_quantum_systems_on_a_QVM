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
from pyquil import paulis
from matplotlib import cm
from pyquil.api import WavefunctionSimulator
from mpl_toolkits.mplot3d import Axes3D

samples = 100
j = 3 / 2
V = 1
i = 1
h = hamiltonian(j, V)[i]
print(h.toarray())
eigvals = eigs(j, V)[i]
print(eigvals)
qc = get_qc('3q-qvm')
H = matrix_to_op.multi_particle(h)
coeffsquare = [(term.coefficient ** 2).real for term in H.terms]
tol = np.sqrt(sum(coeffsquare)) / np.sqrt(samples)

vqe = vqeOverride.VQE_override(minimizer=minimize,
                               minimizer_kwargs={'method': 'Nelder-Mead'})
sweep_steps = 30
parameters = np.linspace(-5, 5, sweep_steps)
print('Tolerance: ', tol)
min_eig = vqe_eig.smallest(H, qc=qc,
                           initial_params=
                           init_params.alternate(
                               h.shape[0]),
                           ansatz_=ansatz.multi_particle(h.shape[0]),
                           disp=True,
                           fatol=tol, xatol=1e-4, samples=samples,
                           return_all_data=True)

print('Min eig vqe: ', min_eig)
optparam = min_eig['x']
min_eig_exp, variance = vqe.expectation(
    ansatz.one_particle_ucc(h.shape[0])(optparam), H,
    samples=10000,
    qc=qc)

sweep_vals = [
    list(vqe.expectation(ansatz.one_particle_ucc(h.shape[0])(np.array([para])),
                         H,
                         samples=samples,
                         qc=qc)) for para in parameters]
exp_val = [returns[0] for returns in sweep_vals]
std_val = [np.sqrt(returns[1]) for returns in sweep_vals]
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
#  initial_params=init_params.one_particle_alt(h.shape[0]),
#  samples=samples, qc=qc, disp=print)
'''

parameters_none = np.linspace(-5, 5, 500)
exp_val2 = [
    vqe.expectation(ansatz.one_particle_ucc(h.shape[0])(np.array([para])), H,
                    samples=None,
                    qc=qc)[0] for para in parameters_none]

plt.figure(0)
errs = [2 * i for i in std_val]
plt.errorbar(parameters, exp_val, yerr=errs, ecolor='red', fmt='bo', ms=1,
             label='Samples: {}'.format(samples), barsabove=True)
plt.plot(parameters_none, exp_val2, 'black', label='Samples: None')
plt.errorbar(optparam, min_eig_exp, yerr=2 * np.sqrt(variance), fmt='g*', ms=15,
             barsabove=True, ecolor='red')
plt.xlabel('Parameter value')
plt.ylabel('Expected value of Hamiltonian')
print('Min eig vqe_exp: ', min_eig_exp, '\nVariance: ', variance)
plt.show()
