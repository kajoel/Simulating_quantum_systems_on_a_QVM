# @author = Carl, 1/5
# Imports
import numpy as np
from constants import ROOT_DIR
from os.path import join
from core import data
from analyze import NM_fel_measmax


def bayes_save(version, size, ansatz_name):
    base_dir = join(ROOT_DIR, f'data/final_bayes/v{version}')
    data_file1 = f'{ansatz_name}_bayes_size={size}_part_1.pkl'
    data_file2 = f'{ansatz_name}_bayes_size={size}_part_2.pkl'

    data1 = data.load(data_file1, base_dir)[0]
    data2 = data.load(data_file2, base_dir)[0]

    data_ = data1 + data2

    fel = np.zeros([36, 60])
    nr = np.zeros([36, 60])

    samples_lst = np.zeros([36])
    max_meas_lst = np.zeros([60])
    identifier_set = set()


    for i, y in enumerate(data_):
        identifier, result = y
        samples = identifier[-1]
        max_meas = identifier[-2]

        if identifier in identifier_set or samples>256500:
            continue

        identifier_set.add(identifier)

        samples_idx = np.argmin(np.abs(np.linspace(2750, 256500, 36) - samples))
        max_meas_idx = np.argmin(np.abs(np.linspace(50000, 3e6, 60) - max_meas))

        eig = result['correct']
        fun = result['fun']

        error = abs((eig - fun) / eig * 100)

        max_meas_lst[max_meas_idx] = max_meas
        samples_lst[samples_idx] = samples

        if 4<int(round(max_meas/samples))<=300:
            fel[samples_idx,max_meas_idx] +=error
            nr[samples_idx,max_meas_idx] +=1

        for i in range(36):
            for j in range(60):
                if nr[i,j]!=0:
                    fel[i,j] /= nr[i,j]


        for i in range(36):
            samples = samples_lst[i]
            tmp = 0
            count = 0
            if samples==0 or max_meas_lst[0]==0:
                continue
            temp_evals = int(round(max_meas_lst[0]/samples))
            for j in range(60):
                max_meas = max_meas_lst[j]
                fun_evals = int(round(max_meas/samples))
                if j==59 and fun_evals == temp_evals:
                    tmp += fel[i, j]
                    count += 1
                    mean = tmp / count
                    fel[i, j - count:j] = mean
                    continue
                if fun_evals == temp_evals:
                    tmp += fel[i,j]
                    count +=1
                else:
                    mean = tmp/count
                    fel[i,j-count-1:j-1] = mean
                    tmp = fel[i,j]
                    count=1
                    temp_evals = fun_evals


    file = f'NM_heatmap/v2/{ansatz_name}_bayes_size={size}.pkl'
    data2 = np.linspace(50000, 3e6, 60), np.linspace(2750, 256500, 36), fel
    data.save(file, data2, extract=True)


def NM_save(version, size, ansatz_name):
    pass


version = 4
size = 3
ansatz_name='multi_particle'

bayes_save(version, size, ansatz_name)