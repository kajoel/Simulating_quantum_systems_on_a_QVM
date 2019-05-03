# Carl 10/4
###############################################################################
# Imports
from core import data
import numpy as np
import matplotlib.pyplot as plt
from constants import ROOT_DIR
from os.path import join
import itertools


###############################################################################

def handle_i(i):
    if i % 3 == 0:
        return 6
    elif i % 3 == 2:
        return 3
    else:
        return 0


base_dir = join(ROOT_DIR, 'data/NelderMead_Restart_v1')

j = 1
samples = np.linspace(500, 10000, 20)
ansatz_types = ['one_particle', 'one_particle_ucc', 'multi_particle',
                'multi_particle_ucc']

x = [np.zeros([20, 9]) for i in range(4)]
fun = [np.zeros([20, 9]) for i in range(4)]
fun_evals = [np.zeros([20, 9]) for i in range(4)]
fel = [np.zeros([20, 9]) for i in range(4)]
len_H = [np.zeros([20, 9]) for i in range(4)]


for sample, k in itertools.product(samples, range(1, 4)):
    a = int(sample / 500)
    sample = int(sample)
    for i in range(12 * (a - 1) + 1, 12 * a + 1):
        file = 'NelderMead_Restart_j={}_samples={}_i={}_k={}.pkl'. \
            format(j, sample, i, k)
        data_, metadata = data.load(file, base_dir)

        for idx in range(4):
            if metadata['ansatz'] == ansatz_types[idx]:
                x[idx][a - 1, handle_i(i) + (k) - 1] = data_['x']
                fun[idx][a - 1, handle_i(i) + (k) - 1] = data_['fun']
                fun_evals[idx][a - 1, handle_i(i) + (k) - 1] = np.sum(
                    data_['fun_evals'])
                fel[idx][a - 1, handle_i(i) + (k) - 1] = np.linalg.norm(
                    data_['facit'] - data_['x'])
                len_H[idx][a - 1, handle_i(i) + (k) - 1] = metadata['len_H']


colors = ['k', 'b', 'r', 'g']
fontsize = 10

fel_mean = [np.zeros([20,3]) for i in range(4)]

for ansatz in range(4):
    for i, row in enumerate(fel[ansatz]):
        fel_mean[ansatz][i, 0] = np.mean(row[0:2])
        fel_mean[ansatz][i, 1] = np.mean(row[3:5])
        fel_mean[ansatz][i, 2] = np.mean(row[6:8])


fun_evals_mean = [np.zeros([20,3]) for i in range(4)]

for ansatz in range(4):
    for i, row in enumerate(fun_evals[ansatz]):
        fun_evals_mean[ansatz][i, 0] = np.mean(row[0:2])
        fun_evals_mean[ansatz][i, 1] = np.mean(row[3:5])
        fun_evals_mean[ansatz][i, 2] = np.mean(row[6:8])

for fig in range(3):
    plt.figure(fig)
    for ansatz in range(4):
        plt.scatter(samples * len_H[ansatz][:, fig * 3] * fun_evals_mean[ansatz][:, fig],
                    fel_mean[ansatz][:, fig], color=colors[ansatz],
                    label=ansatz_types[ansatz])

    plt.title('j = 1, Size of matrix = 2, Max values at same parameter: {}\
    \n(Mean of 3 measurements)'.format(fig+3), fontsize=fontsize)
    plt.xlabel('Antal mätning', fontsize=fontsize)
    plt.ylabel('Parameterfel', fontsize=fontsize)
    plt.xticks(fontsize=fontsize-2)
    plt.yticks(fontsize=fontsize-2)
    plt.ticklabel_format(useOffset=False, style='plain')
    plt.legend()
    #plt.xlim(0, 100000)

plt.show()
