from core.interface import *
from core import vqe_eig
from core import init_params
import numpy as np

size = 3
ansatz_name = 'one_particle_ucc'
h, eig = hamiltonians_of_size(size)
initial_params = init_params.zeros(h[0].shape[0])
initial_params = np.array([1,0])
H, qc, ansatz_, _ = create_and_convert(ansatz_name,h[0])
samples = 100000
vqe = vqe_nelder_mead(samples=samples, H=H)

print(eig[0])

result = vqe_eig.smallest(H, qc,initial_params, vqe,ansatz_, samples = samples,
                          return_all=True, disp=True, max_meas=samples*20)

print(result)

