from lipkin_quasi_spin import hamiltonian, eigs
from grove.pyvqe.vqe import VQE
import numpy as np
from pyquil import get_qc
from scipy.optimize import minimize
from ansatz import one_particle
import matrix_to_op
import matplotlib.pyplot as plt
from matplotlib import cm
import init_params
from pyquil.api import WavefunctionSimulator
from mpl_toolkits.mplot3d import Axes3D

samples = None
j = 3
V = 1
h = hamiltonian(j, V)[1]
print(h.toarray())
eigvals = eigs(j, V)[1]
print(eigvals)
qc = get_qc('3q-qvm')
H = matrix_to_op.one_particle(h)
vqe = VQE(minimizer=minimize, minimizer_kwargs={'method': 'Nelder-Mead'})
sweep_steps = 20
parameters = np.linspace(-5, 5, sweep_steps)

#exp_val = [vqe.expectation(one_particle(np.array([para])), H, samples=None,
                          # qc=qc) for para in parameters]

exp_val = np.zeros((sweep_steps, sweep_steps))
mesh_1 = np.zeros((sweep_steps, sweep_steps))
mesh_2 = np.zeros((sweep_steps, sweep_steps))

for i, p_1 in enumerate(parameters):
    mesh_1[i] += p_1
    mesh_2[i] = parameters
    exp_val[i] = [vqe.expectation(one_particle(np.array([p_1, p_2])), H,
                                  samples=samples, qc=qc)
                  for p_2 in parameters]
    fig = plt.figure(0)
    ax = fig.add_subplot(111, projection='3d')
    # Plot the surface

    #save_run_to_csv(exp_val)
    ax.plot_surface(mesh_1, mesh_2, exp_val, cmap=cm.coolwarm)

'''
min_eig = vqe.vqe_run(one_particle, H,
                      initial_params=init_params.one_particle_alt(h.shape[0]),
                      samples=samples, qc=qc, disp=print)

print('Min eig vqe: ', min_eig['fun'])
'''
# exp_val2 = [vqe.expectation(one_particle(np.array([para])), H, samples=samples,
# qc=qc) for para in parameters]

#plt.figure(0)
#plt.plot(parameters, exp_val, 'black', label='Samples: None')
# plt.plot(parameters, exp_val2, 'red', label='Samples: {}'.format(samples))
#plt.xlabel('Parameter value')
#plt.ylabel('Expected value of Hamiltonian')
plt.show()
