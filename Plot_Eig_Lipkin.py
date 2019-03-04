from quasi_spin_hamiltonian import quasi_spin_hamiltonian
from irrep_multiplicity import irrep_multiplicity
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.sparse as sparse

tol = 10e-10

N=9
V=0

(j, cj) = irrep_multiplicity(N)

H_j = {0: [0]}
map_eig = {0: 0}

(j, cj) = irrep_multiplicity(N)

eig1 = np.empty((0,2), float)
eig2 = np.empty((0,3), float)

V = np.linspace(0,1,100)
for v in V:
    i=N/2
    if i!=0:
        (H1, H2) = quasi_spin_hamiltonian(i, v)
        H1_eig = np.linalg.eigh(H1.todense())[0]
        H2_eig = np.linalg.eigh(H2.todense())[0]
        h_arr = np.empty((1, 0), float)
        for h in H1_eig:
            if h>tol: h_arr = np.append(h_arr,h)
        eig1 = np.append(eig1, [h_arr],axis=0)
        h_arr = np.empty((1, 0), float)
        for h in H2_eig:
            if h>tol: h_arr = np.append(h_arr,h)
        eig2 = np.append(eig2, [h_arr], axis=0)
plt.plot(V,eig1,linewidth=1.2)
plt.plot(V,eig2,linewidth=1.2)
plt.show()