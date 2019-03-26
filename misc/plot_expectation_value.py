# Carl 12/3
###############################################################################
<<<<<<< HEAD
#Imports
from pyquil import Program, get_qc
import time
=======
# Imports
from core.lipkin_quasi_spin import hamiltonian, eigs
from grove.pyvqe.vqe import VQE
>>>>>>> develop
import numpy as np
from scipy.optimize import minimize
<<<<<<< HEAD
from grove.pyvqe.vqe import VQE
from matplotlib import pyplot as plt
from datetime import datetime,date
from matplotlib import cm

# Imports from our projects
from matrix_to_operator import matrix_to_operator_1
from lipkin_quasi_spin import hamiltonian,eigs
from ansatz import one_particle as ansatz
from ansatz import one_particle_inital, carls_initial
from vqe_eig import smallest as vqe_eig

###############################################################################
def sweep_parameters(H, qvm_qc, new_version=False, num_para=20, start=-10,
                     stop=10, samples=None, fig_nr=0, save=False):
    '''
    TODO: Add a statement that saves the data from the run and comments.
    '''
    vqe = VQE(minimizer=minimize, minimizer_kwargs={'method': 'Nelder-Mead'})
    if H.shape[0] > 3:
        print('To many parameters to represent in 2 or 3 dimensions')
        return
    elif H.shape[0] is 1:
        print('Nothing to sweep over')
        return
    elif H.shape[0] is 2:
        H = matrix_to_operator_1(H)
        parameters = np.linspace(start, stop, num_para)

        if new_version:
            exp_val = [vqe.expectation(ansatz(np.array([para])), H,
                                       samples=samples, qc=qvm_qc)
                       for para in parameters]
        else:
            exp_val = [vqe.expectation(ansatz(np.array([para])), H,
                                       samples=samples, qvm=qvm_qc)
                       for para in parameters]

        plt.figure(fig_nr)
        plt.plot(parameters, exp_val, label='Samples: {}'.format(samples))
        plt.xlabel('Paramter value')
        plt.ylabel('Expected value of Hamiltonian')
        return
    else:
        H = matrix_to_operator_1(H)
        exp_val = np.zeros((num_para, num_para))
        mesh_1 = np.zeros((num_para, num_para))
        mesh_2 = np.zeros((num_para, num_para))
        parameters = np.linspace(start, stop, num_para)

        for i, p_1 in enumerate(parameters):
            mesh_1[i] += p_1
            mesh_2[i] = parameters
            if new_version:
                exp_val[i] = [vqe.expectation(ansatz(np.array([p_1, p_2])), H,
                                              samples=samples, qc=qvm_qc)
                              for p_2 in parameters]
            else:
                exp_val[i] = [vqe.expectation(ansatz(np.array([p_1, p_2])), H,
                                              samples=samples, qvm=qvm_qc)
                              for p_2 in parameters]

        fig = plt.figure(fig_nr)
        ax = fig.add_subplot(111, projection='3d')
        # Plot the surface

        save_run_to_csv(exp_val)
        ax.plot_surface(mesh_1, mesh_2, exp_val, cmap=cm.coolwarm)
        return

###############################################################################
def save_run_to_csv(Variable):
    """ Save given variable to CSV-file with name of exact time
    Arguments:
        Variable{np.array}--Variable to save to .txt file in CSV-format
    """
    np.savetxt('{}'.format(datetime.now()), Variable)
###############################################################################
j = 3
V = 1
h = hamiltonian(j, V)
h = h[0]
qc = get_qc(str(h.shape[0]) + 'q-qvm')
###############################################################################
<<<<<<< HEAD
eig_CPU = eigs(j, V)
print(eig_CPU[0])
#eig_CPU = eig_CPU[1]
#start = time.time()
#end = time.time()
#print('Tid VQE: ' + str(end-start))

vqe = VQE(minimizer=minimize, minimizer_kwargs={'method': 'Nelder-Mead'})
H = matrix_to_operator_1(h)
eig = 0
#for i in range(50):
    #eig += vqe.expectation(ansatz(result['x']), H, samples=20000, qc=qc)

#eig = eig/50
#print(eig)
#plt.plot(result['x'],eig,'*', ms=10)
#plt.show()
=======
H = matrix_to_op.multi_particle(h)
initial_params = init_params.alternate(h.shape[0])
vqe_eig.smallest(H, qc, initial_params, ansatz_=ansatz.multi_particle,
                 samples=samples, fatol=1e-2, disp_run_info = True)
>>>>>>> develop
###############################################################################
result = vqe_eig(h, qc_qvm=qc, initial=carls_initial(h.shape[0]),
                 num_samples=None, disp_run_info=True, display_after_run=True,
                 xatol=1e-3, fatol=5e-1, return_all_data=True)
for i in range(20):
    eig += vqe.expectation(ansatz(result['x']), H, samples=500, qc=qc)

eig = eig/20
print(result['fun'])
print(eig)