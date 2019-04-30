from core.interface import *
from core import vqe_eig
from core import init_params
import numpy as np

x = []
norm = []
matrix = 0
ansatz_name = 'one_particle_ucc'

for size in range(2,7):
    h, eig = hamiltonians_of_size(size)
    initial_params = init_params.ucc(h[matrix].shape[0])
    print(initial_params)
    H, qc, ansatz_, _ = create_and_convert(ansatz_name,h[matrix])
    #print(H)
    samples = 10000
    vqe = vqe_nelder_mead(fatol=1e-3)

    #print(ansatz_(initial_params))
    #print(eig[matrix])

    result = vqe_eig.smallest(H, qc,initial_params, vqe,ansatz_, samples = samples,
                              return_all=True, disp=True, max_meas=samples*20)

    x.append(result['x'])
    norm.append(np.linalg.norm(result['x']))

for x_, norm_ in zip(x,norm):
    print(x_)
    print(norm_)