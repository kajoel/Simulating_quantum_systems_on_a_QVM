# @author = Axel, 9/5
# Imports
import numpy as np
from constants import ROOT_DIR
from os.path import join
from core import data



def bayes_save_to_csv(version, ansatz_name, size=3, minimizer='bayes'):
    base_dir = join(ROOT_DIR, f'data/final_bayes/v{version}')
    data_file1 = f'{ansatz_name}_{minimizer}_size={size}_part_{1}.pkl'
    data_file2 = f'{ansatz_name}_{minimizer}_size={size}_part_{2}.pkl'

    data_1 = data.load(data_file1, base_dir)[0]
    data_2 = data.load(data_file2, base_dir)[0]
    data_ = data_1+data_2

    fel = np.zeros([36, 60])
    samples_lst = []
    max_meas_lst = []
    nr = np.zeros([36, 60])

    for i, y in enumerate(data_):
        identifier, result = y
        samples = identifier[6]
        max_meas = identifier[5]
        arr = np.asarray(np.abs(np.linspace(2750, 256500, 36) - samples))
        samples_idx = np.argmin(arr)
        max_meas_idx = np.argmin(np.abs(np.linspace(50000, 3e6, 60) - max_meas))

        eig = result['correct']
        fun_none = result['fun']

        error = (eig - fun_none) / eig * 100

        max_meas_lst.append(max_meas)
        samples_lst.append(samples)

        fel[samples_idx][max_meas_idx] +=error
        nr[samples_idx][max_meas_idx] +=1

    for j in range(36):
        for k in range(60):
            if np.any(nr[j,k] != 0):
                fel[j,k]/=nr[j,k]
            else:
                fel[j,k] = 0


    
    import csv
    vec1, vec2, vec3 = [],[],[]

    for i,sample in enumerate(np.linspace(2750, 256500, 36)):
        for j, evals in enumerate(np.linspace(50000, 3e6, 60)):
            vec1.append(evals)
            vec2.append(sample)
            if fel[i,j] == 0.0: vec3.append('nan')
            else: vec3.append(np.log10(fel[i,j]))
            
                
        with open('text.csv', 'w') as f:
            writer = csv.writer(f, delimiter=' ')    
            writer.writerows(zip(vec1, vec2, vec3))
        
   

if __name__ == '__main__':
    version = 4
    size = 3
    ansatz_name = 'multi_particle'
    minimizer = 'bayes'

    bayes_save(version, ansatz_name)




