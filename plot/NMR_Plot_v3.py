# Carl 14/4
###############################################################################
# Imports
from core import data
import numpy as np
import matplotlib.pyplot as plt
from constants import ROOT_DIR
from os.path import join
import itertools
from core.lipkin_quasi_spin import hamiltonian
from core import init_params
from plot import tikzfigure

###############################################################################

fontsize = 10

base_dir = join(ROOT_DIR, 'data/NelderMead_Restart_v2')

j, V, matrix = 1, 1, 0
ansatz_types = ['one_particle_ucc', 'multi_particle']
#ansatz_types = ['one_particle_ucc']
file = 'NMR_j={}.pkl'.format(j)
data_, metadata = data.load(file, base_dir)

colors = ['k', 'r']

labels = ['Enpartikel-UCC', 'Flerpartikel-ASG']

for fig, max_param in zip(range(5), range(5,10)):
    print(fig)
    plt.figure(fig)
    for i, ansatz in enumerate(ansatz_types):
        sample, fun_evals, fel = np.array([]), np.array([]), np.array([])


        for meas in data_[ansatz]:
            sample_mean, fun_evals_mean, fel_mean = [], [], []
            for iter in range(1,6):
                try:
                    result = meas['max_param={}'.format(max_param)]\
                    ['iter={}'.format(iter)]
                except KeyError:
                    break

                sample_mean.append(result['samples'])
                fun_evals_mean.append(np.sum(result['fun_evals']))
                fel_mean.append(np.linalg.norm(result['x'] - result['facit']))

            sample = np.append(sample, np.mean(sample_mean))
            fun_evals = np.append(fun_evals, np.mean(fun_evals_mean))
            fel = np.append(fel, np.mean(fel_mean))

        plt.scatter(sample * fun_evals, fel, s=3, color=colors[i], label=labels[i])

    plt.title('j = {}, Size of matrix = {}, Max values at same parameter: {}\
    \n(Mean of 5 measurements)'.format(j, j+1, max_param), fontsize=fontsize)
    plt.xlabel('Antal mätning', fontsize=fontsize)
    plt.ylabel('Parameterfel', fontsize=fontsize)
    plt.xticks(fontsize=fontsize-2)
    plt.yticks(fontsize=fontsize-2)
    plt.legend()
    #plt.ylim(-0.01, 0.1)


#tikzfigure.save('NMR_j=1')
plt.show()
