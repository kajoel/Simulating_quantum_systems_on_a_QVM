# Carl 25/4
###############################################################################
# Imports
from core import data
import numpy as np
import matplotlib.pyplot as plt
from constants import ROOT_DIR
from os.path import join
import itertools
from core.lipkin_quasi_spin import hamiltonian, eigs
from core import init_params
from plot import tikzfigure
from core import ansatz
from core.interface import create, hamiltonians_of_size, nelder_mead


size = 2

base_dir = join(ROOT_DIR, 'data/NM_Restart_Parallel_v1/90591188723')
file2 = 'NM_Restart_Parallel_v1/fun_none/v1_multi_particle_size=2.pkl'

color_map = np.array(
    [(161, 218, 180), (65, 182, 196), (34, 94, 168), (37, 52, 148)]) / 255

nr_pts = {}

for mat_idx in range(4):
    file = f'v1_multi_particle_size={size}_matidx={mat_idx}.pkl'
    data_, metadata = data.load(file, base_dir)
    eig, fun_none = data.load(file2)[0][0][mat_idx]

    fel = {3: [], 4: [], 5: [], 6: [], 7:[], 8: [], 9: []}

    for max_para in range(3,10):
        for fun_none_ in fun_none[max_para]:
            fel[max_para].append(np.abs((fun_none_-eig)/(-eig))*100)

    meas = {3: [], 4: [], 5: [], 6: [], 7:[], 8: [], 9: []}

    for i, tmp in enumerate(data_):
        identifier, result = tmp
        max_para = identifier[4]
        meas[max_para].append(identifier[3] * result['fun_evals'])

    nr_tmp = {}
    for max_para in range(3, 10):
        nr_tmp[max_para] = 0
        for meas_, fel_ in zip(meas[max_para],fel[max_para]):
            if meas_<100000 and fel_<1:
                nr_tmp[max_para] +=1

    nr_pts[mat_idx] = nr_tmp

    '''
    for fig in range(3,10):
        label = f'Matrix index: {mat_idx}'
        plt.figure(fig-2)
        plt.plot(meas[fig],fel[fig], 'o',markersize=2, label=label, color=color_map[mat_idx])
        plt.title(f'Maximum number of same parameters: {fig}')
        plt.xlabel('Antel mätningar')
        plt.ylabel('Procentuellt fel')
        plt.legend()
        #plt.xlim(-100,1000000)
    '''

print('Number of points:')
tot_pts = [0 for i in range(7)]
for matrix, max_params in zip(nr_pts.keys(), nr_pts.values()):
    print('')
    for max_param, pts in zip(max_params.keys(), max_params.values()):
        print(f'Matrix: {matrix}, Max_param: {max_param}, Points = {pts}')
        tot_pts[max_param-3] += pts

print('\nTotal number of points:')

print(tot_pts)

plt.show()
