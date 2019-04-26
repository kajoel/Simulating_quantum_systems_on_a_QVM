# Carl 25/4
###############################################################################
# Imports
from core import data
import numpy as np
import matplotlib.pyplot as plt
from constants import ROOT_DIR
from os.path import join
import itertools
from plot import tikzfigure


color_map = np.array(
    [(161, 218, 180), (65, 182, 196), (34, 94, 168), (37, 52, 148)]) / 255


def tot_pts(meas_tol, fel, meas, fel_tol=1,):
    nr_tot = [0 for i in range(9)]
    for mat_idx in range(4):
        for max_para in range(1, 10):
            for meas_, fel_ in zip(meas[mat_idx][max_para],fel[mat_idx][max_para]):
                if meas_<meas_tol and fel_<fel_tol:
                    nr_tot[max_para-1] +=1

    return nr_tot


def plot_meas_fel(meas, fel, fig=1):
    plt.figure(fig)
    for mat_idx in range(4):
        label = f'Matrix index: {mat_idx}'

        for fig in range(1,10):
            plt.subplot(3,3,fig)
            plt.plot(meas[mat_idx][fig],fel[mat_idx][fig], 'o',markersize=2,
                     label=label, color=color_map[mat_idx])
            plt.title(f'Maximum number of same parameters: {fig}')
            #plt.xlabel('Antel mätningar')
            #plt.ylabel('Procentuellt fel')
            plt.legend()
            plt.xlim(-100,3000000)


size = 2
version = 2

base_dir = join(ROOT_DIR, f'data/NM_Restart_Parallel_v{version}')



nr_pts = {}

fel = {mat_idx: {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}\
       for mat_idx in range(4)}

meas = {mat_idx: {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}\
       for mat_idx in range(4)}

for mat_idx in range(4):
    file = f'multi_particle_size={size}_matidx={mat_idx}.pkl'
    data_, metadata = data.load(file, base_dir)

    if version == 1:
        file2 = 'NM_Restart_Parallel_v1/fun_none/v1_multi_particle_size=2.pkl'
        eig, fun_none = data.load(file2)[0][0][mat_idx]
        for max_para in range(3,10):
            for fun_none_ in fun_none[max_para]:
                fel[max_para].append(np.abs((fun_none_-eig)/(-eig))*100)


    for i, tmp in enumerate(data_):
        identifier, result = tmp
        max_para = identifier[4]
        meas[mat_idx][max_para].append(identifier[3] * result['fun_evals'])
        fel[mat_idx][max_para].append(np.abs(result['fun_none']-result['correct'])*100)


plot_meas_fel(meas, fel)

#tot_pts_tmp = np.asarray([])
tot_pts_tmp = []
meas_tol = np.arange(10000, 3000000, 10000)
for meas_tol_ in meas_tol:
    #tot_pts_tmp = np.append(tot_pts_tmp, np.asarray(tot_pts(meas_tol_, fel, meas)))
    tot_pts_tmp.append(tot_pts(meas_tol_, fel, meas, fel_tol=1))

plt.figure(2)
legend = [f'{i}' for i in range(1,10)]
plt.plot(meas_tol, tot_pts_tmp)
plt.legend(legend)
plt.show()

'''
nr_tmp = pts_in_square(100000, fel, meas)
nr_pts[mat_idx] = nr_tmp





#print('Number of points:')
tot_pts = [0 for i in range(9)]
for matrix, max_params in zip(nr_pts.keys(), nr_pts.values()):
    #print('')
    for max_param, pts in zip(max_params.keys(), max_params.values()):
        #print(f'Matrix: {matrix}, Max_param: {max_param}, Points = {pts}')
        tot_pts[max_param-1] += pts

print('\nTotal number of points:')

print(tot_pts)


plt.show()
'''