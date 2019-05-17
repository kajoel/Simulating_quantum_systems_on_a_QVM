from core.interface import hamiltonians_of_size
from scipy.sparse.linalg import eigsh
import numpy as np


for size in range(2, 5):
    print(f'\nsize = {size}')

    h_lst = hamiltonians_of_size(size, V=1, e=0)[0]

    for h in h_lst:
        eig_vect = eigsh(h, k=1, which='SA')[1]

        conjecture = np.array([(-1)**i for i in range(size)])/np.sqrt(size)

        if eig_vect[0] < 0:
            eig_vect *= -1

        print(np.linalg.norm(eig_vect.transpose() - conjecture))

