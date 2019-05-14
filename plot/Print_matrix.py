# @author = Carl, 12/5
# Imports
import numpy as np
from core.lipkin_quasi_spin import hamiltonian

for j in np.arange(1, 7, 0.5):
    for matrix in [0,1]:
        h = hamiltonian(j, 1)[matrix]
        h = h.todense()
        h = np.array(h)
        if h.shape[0]>5:
            continue

        print('\\begin{equation}')
        print(f'\tH_{j}^{matrix+1} = ')
        print('\t\\begin{bmatrix}')

        for k, row in enumerate(h):
            h_str = ''
            for i, hij in enumerate(row):
                V_e = ['\\epsilon', 'V'][i!=k]
                if hij!=0 and hij!=float(int(hij)) and abs(hij**2 - int(hij**2))<1e3:
                    hij_str = '\\sqrt{' + str(int(round(hij**2))) + '}'
                elif hij==float(int(hij)):
                    hij_str = str(int(hij))
                else:
                    hij_str = str(hij)
                if i<len(row)-1:
                    h_str += hij_str + V_e + '&\t'
                else:
                    h_str += hij_str + V_e + '\\\\'
            print(f'\t\t{h_str}')

        print('\t\\end{bmatrix}')
        print('\\end{equation}')