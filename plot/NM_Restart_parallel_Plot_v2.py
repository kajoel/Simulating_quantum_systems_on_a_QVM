# Carl 28/4
###############################################################################
# Imports
from core import data
import numpy as np
import matplotlib.pyplot as plt
from constants import ROOT_DIR
from os.path import join
import itertools
from plot import tikzfigure


def meas_fel(version, size):
    base_dir = join(ROOT_DIR, f'data/NM_Restart_Parallel/v{version}')
    meas, fel, status = {}, {}, {}

    for mat_idx in range(4):
        file = f'multi_particle_size={size}_matidx={mat_idx}.pkl'
        data_, _ = data.load(file, base_dir)

        # meas_tmp = {}
        # fel_tmp = {}
        # status_tmp = {}

        for i, x in enumerate(data_):
            identifier, result = x

            samples = identifier[3]
            if version == 5:
                max_para = 0
                repeats = identifier[4]

            else:
                max_para = identifier[4]
                repeats = identifier[5]

            eig = result['correct']
            fun_none = result['fun_none']
            fun_evals = result['fun_evals']

            try:
                meas[max_para]
            except:
                meas[max_para] = []
                fel[max_para] = []
                status[max_para] = []
            if repeats <= 1:
                meas[max_para].append(samples * fun_evals)
                fel[max_para].append(np.abs((fun_none - eig) / eig) * 100)
                status[max_para].append(result['status'])

        # meas[mat_idx], fel[mat_idx], status[mat_idx] = meas_tmp, fel_tmp, status_tmp

    return meas, fel, status


def tot_pts(meas, fel, meas_tol, fel_tol=1):
    l = len(meas.keys())
    nr_tot = [0 for i in range(l)]
    for max_para in meas.keys():
        for meas_, fel_ in zip(meas[max_para], fel[max_para]):
            if meas_ < meas_tol and fel_ < fel_tol:
                nr_tot[max_para - list(meas.keys())[0]] += 1

    for i, max_para in enumerate(meas.keys()):
        nr_tot[i] /= len(meas[max_para])
    return nr_tot


def plot_data(fel, meas, status, max_para=None):
    color_map = np.array(
        [(161, 218, 180), (65, 182, 196), (34, 94, 168), (37, 52, 148)]) / 255

    l = len(meas.keys())
    if max_para == None:
        ax = [plt.subplot(l, 1, i) for i in range(1, l + 1)]
    # [exec(f'ax{i}=plt.subplot({l+1},1,{i})') for i in range(1,l+1)]

    for mat_idx in range(4):
        label = f'Matrix index: {mat_idx}'
        if max_para == None:
            for i, max_para_ in enumerate(meas[mat_idx].keys()):
                for meas_, fel_, status_ in zip(meas[mat_idx][max_para_],
                                                fel[mat_idx][max_para_],
                                                status[mat_idx][max_para_]):
                    print(i)
                    if status_ == 1:
                        mark_idx = 1
                    elif status_ == -1:
                        mark_idx = 2
                    else:
                        mark_idx = 0
                    marker = ['o', '^', 'v']
                    ax[i].scatter(meas[mat_idx][max_para_],
                                  fel[mat_idx][max_para_],
                                  marker=marker[mark_idx],
                                  linewidth=.5,
                                  color=color_map[mat_idx])
                ax[i].legend(['0', '1', '2', '3'])
                ax[i].set_title(f'Maxpara: {max_para_}')

        else:
            plt.plot(meas[mat_idx][max_para], fel[mat_idx][max_para], 'o',
                     markersize=2, linestyle='None', label=label,
                     color=color_map[mat_idx])
            # plt.title(f'Maxpara: {max_para}')
            plt.legend()


def plot_pts(meas, fel, fel_tol, min_meas, max_meas):
    tot_pts_tmp = []
    meas_tol_range = np.arange(min_meas, max_meas, 1000)
    for meas_tol in meas_tol_range:
        tot_pts_tmp.append(tot_pts(meas, fel, meas_tol, fel_tol))

    for i in meas.keys():
        label = f'Maxpara={i}'
        k = i - list(meas.keys())[0]
        plt.plot(meas_tol_range, np.asarray(tot_pts_tmp)[:, k], label=label)


def plot_data_v2(meas, fel, status):
    marker = ['o', '^', 'v']
    color_map = ['k', 'r', 'b']
    markersize=[2, 10, 10]
    label=[f'Status={0}', f'Status={1}', f'Status={-1}']

    l = len(meas.keys())
    print(l)
    if l > 1:
        fig, ax = plt.subplots(2, 2)
        ax = ax.flatten()
    else:
        fig, ax = plt.subplots(1, 1)
        ax = [ax]

    for i, max_para in enumerate(meas.keys()):
        p = []
        legend_text = []
        tmp = set()
        for meas_, fel_, status_ in zip(meas[max_para], fel[max_para],
                                        status[max_para]):
            if status_ == 1:
                mark_idx = 1
            elif status_ == -1:
                mark_idx = 2
            else:
                mark_idx = 0
            if mark_idx in tmp:
                ax[i].plot(meas_, fel_, marker=marker[mark_idx],
                           markersize=markersize[mark_idx],
                           color=color_map[mark_idx],
                           linestyle='None')
            else:
                tmp.add(mark_idx)
                p.extend(ax[i].plot(meas_, fel_, marker=marker[mark_idx],
                           markersize=markersize[mark_idx],
                           color=color_map[mark_idx],
                           linestyle='None'))
                legend_text.append(status_)



        ax[i].set_title(f'Maxpara: {max_para}')
        ax[i].legend(p, legend_text)

fel_tol = 1
min_meas = 1e6
max_meas = 3e6

size = 3
max_para = 6

meas_v6, fel_v6, status_v6 = meas_fel(6, size)
meas_v4, fel_v4, status_v4 = meas_fel(4, size)
meas_v5, fel_v5, status_v5 = meas_fel(5, size)

#plot_pts(meas_v5, fel_v5, fel_tol, min_meas, max_meas)
#plot_pts(meas_v4, fel_v4, fel_tol, min_meas, max_meas)
# plot_pts(meas, fel, fel_tol, min_meas, max_meas)

#plt.figure(2)
#plot_data_v2(meas_v4, fel_v4, status_v4)


plot_data_v2(meas_v5, fel_v5, status_v5)
plt.title(f'Matrix size = {size}, Version={5}, Maxpara={max_para}')
#plt.xlim(0,4000000)
#print(status_v5[0])
for meas, fel, status in zip(meas_v5[0], fel_v5[0], status_v5[0]):
    if int(status) ==0:
        print(f'{meas}\t{fel}')


# plt.figure(2)
# plot_data(fel, meas, max_para = max_para)
# plt.title(f'Matrix size = {size}, Version={4}, Maxpara={max_para}')
# plt.xlim(0,4000000)

# plt.figure(3)
# plot_data(fel, meas)
# plt.title(f'Matrix size = {size}, Version={5}')
# plt.xlim(0,4000000)

#plt.legend()
#tikzfigure.save('NM_stop_reason')
plt.show()
