import numpy as np
from functools import lru_cache

from pyquil import get_qc
from scipy.optimize import minimize
from skopt import gp_minimize

from core import matrix_to_op, init_params, vqe_override
from core.ansatz import one_particle, one_particle_ucc, \
    multi_particle_stereographic, multi_particle_ucc
from core.lipkin_quasi_spin import hamiltonian
from scipy.sparse.linalg import eigsh


def create_and_convert(ansatz_name, h, initial_params=None):
    """
    Creates anstaz, qc and initial_params and converts hamiltonian
    matrix h to hamiltonian PauliSum H.

    @author: Carl

    :param ansatz_name:
    :param h:
    :param initial_params:
    :return:
    """
    dim = h.shape[0]

    if ansatz_name == 'one_particle':
        qc = get_qc(str(h.shape[0]) + 'q-qvm')
        H = matrix_to_op.one_particle(h)
        ansatz_ = one_particle(h)
        if initial_params == None:
            initial_params = init_params.alternate(dim)

    elif ansatz_name == 'one_particle_ucc':
        qc = get_qc(str(h.shape[0]) + 'q-qvm')
        H = matrix_to_op.one_particle(h)
        ansatz_ = one_particle_ucc(h)
        if initial_params == None:
            initial_params = init_params.ucc(dim)

    elif ansatz_name == 'multi_particle':
        qc = get_qc(str(int.bit_length(h.shape[0])) + 'q-qvm')
        H = matrix_to_op.multi_particle(h)
        ansatz_ = multi_particle_stereographic(h)
        if initial_params == None:
            initial_params = init_params.alternate_stereographic(h)

    elif ansatz_name == 'multi_particle_ucc':
        qc = get_qc(str(int.bit_length(h.shape[0])) + 'q-qvm')
        H = matrix_to_op.multi_particle(h)
        ansatz_ = multi_particle_ucc(h)
        if initial_params == None:
            initial_params = init_params.ucc(dim)

    else:
        H, qc, ansatz_ = None, None, None

    return H, qc, ansatz_, initial_params


def smallest_eig(h):
    """
    Find smallest eigenvalue of subscriptable sparse matrix.

    @author = Joel

    :param h: sparse matrix
    :return: smallest eigenvalue
    """
    if h.shape[0] == 1:
        return h[0, 0]
    else:
        return eigsh(h, k=1, which='SA', return_eigenvectors=False)[0]


@lru_cache(maxsize=1)
def hamiltonians_of_size(size: int, V=1., e=1.) -> tuple:
    """

    :param size:
    :param V:
    :param e:
    :return:
    """
    mats = (
        hamiltonian(size - 1, V, e)[0],
        hamiltonian(size - 1 / 2, V, e)[0],
        hamiltonian(size - 1 / 2, V, e)[1],
        hamiltonian(size, V, e)[1]
    )
    eigs = (
        smallest_eig(mats[0]),
        smallest_eig(mats[1]),
        smallest_eig(mats[2]),
        smallest_eig(mats[3])
    )
    return mats, eigs


def vqe_nelder_mead(xatol=1e-2, fatol=None, samples=None, H=None, return_all=False,
                    maxiter=10000):
    '''
    If fatol=None, samples and H must be provided

    @author: Sebastian, Carl, Joel

    :param xatol:
    :param fatol:
    :param return_all:
    :param maxiter:
    :return:
    '''

    if fatol == None:
        fatol = sum(np.abs(term.coefficient) for term in H.terms) / np.sqrt(
            samples)

    disp_options = {'disp': False, 'xatol': xatol, 'fatol': fatol,
                    'maxiter': maxiter, 'return_all': return_all}

    return vqe_override.VQE_override(minimizer=minimize,
                                     minimizer_kwargs={'method': 'Nelder-Mead',
                                                       'options': disp_options})


def vqe_bayes(acq_func="gp_hedge",
              n_calls=15,
              n_random_starts=4,
              random_state=123,
              n_jobs=1):
    '''
    @author: Axel
    :param acq_func: Function to minimize over the gaussian prior.
    :param n_calls: Number of calls to `func`
    :param n_random_starts: Number of evaluations of `func` with random points
                            before approximating it with `base_estimator`.
    :random_state: Set random state to something other than None for
                   reproducible results.
    :param n_jobs: Number of cores to run in parallel while running optimization
    :return:
    '''

    opt_options = {'acq_func': acq_func,
                   'n_calls': n_calls,
                   'n_random_starts': n_random_starts,
                   'random_state': random_state,
                   'n_jobs': n_jobs}

    return vqe_override.VQE_override(minimizer=gp_minimize,
                                     minimizer_kwargs=opt_options)
