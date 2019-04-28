import core.interface
from core.interface import hamiltonians_of_size, vqe_nelder_mead
from core.vqe_override import BreakError
from core import vqe_eig


ansatz_name = "multi_particle"
samples = 5000
max_same_para = 6
h = hamiltonians_of_size(2, 1)[0][0]

H, qc, ansatz_, initial_params = core.interface.create_and_convert(ansatz_name,
                                                                   h)
vqe = vqe_nelder_mead(samples=samples, H=H)
tol_para = 1e-3


def callback(*args, **kwargs):
    raise BreakError


max_fun_evals = 10
result = vqe_eig.smallest(H, qc, initial_params, vqe,
                          ansatz_, samples,
                          callback=callback, max_fun_evals=max_fun_evals)

print('\nThis happens in case of a BreakError:')
print(f'Status: {result.status}')
print(result.message)


def callback(*args, **kwargs):
    pass


max_fun_evals = 3
result = vqe_eig.smallest(H, qc, initial_params, vqe,
                          ansatz_, samples,
                          callback=callback, max_fun_evals=max_fun_evals)

print('\nThis happens when exceeding max_fun_evals')
print(f'Status: {result.status}')
print(result.message)
