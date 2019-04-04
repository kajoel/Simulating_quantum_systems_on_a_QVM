"""
Created on Wed Mar  6 16:35:25 2019
"""

import numpy as np
# Imports for VQE
from scipy.optimize import minimize
from core import ansatz, vqeOverride
from core import init_params
from core import matrix_to_op


def smallest(H, qc, initial_params,
             ansatz_=None,
             samples=None,
             opt_algorithm='Nelder-Mead',
             maxiter=10000,
             disp_run_info=True,
             display_after_run=False,
             xatol=1e-2, fatol=1e-3,
             return_all_data=False,
             # Varför har vi detta som in-argument? Används inte
             convert_op=matrix_to_op.multi_particle,
             print_option=None):
    """
    TODO: Fix this documentation. Below is not up to date.

    Finds the smallest eigenvalue and corresponding -vector of H using VQE.

    @author: Eric, Axel, Carl

    :param H: PauliSum of hamiltonian
    :param qc: either qc or qvm object, depending on version
    :param ansatz_: ansatz function
    :param num_samples: number of samples on the qvm
    :param opt_algorithm:
    :param initial_params: ansatz parameters
    :return: list of energies
    """

    if ansatz_ is None:
        ansatz_ = ansatz.multi_particle

    # All options to Nelder-Mead
    disp_options = {'disp': display_after_run, 'xatol': xatol, 'fatol': fatol,
                    'maxiter': maxiter}

    vqe = vqeOverride.VQE_override(minimizer=minimize,
                                   minimizer_kwargs={'method':
                                                         opt_algorithm,
                                                     'options': disp_options})
    # If disp_run_info is True we will print every step of the Nelder-Mead

    print('Initial parameter:', initial_params, '\n')
    eig = vqe.vqe_run(ansatz_, H, initial_params, samples=samples, qc=qc,
                      disp=disp_run_info, return_all=True)

    # If option return_all_data is True we return a dict with data from all runs
    if return_all_data:
        return eig
    else:
        eigval = eig['fun']
        optparam = eig['x']
        return eigval, optparam


def smallest_restart(H, qc, initial_params,
             ansatz_=None,
             samples=None,
             opt_algorithm='Nelder-Mead',
             maxiter=10000,
             display_after_run=False,
             xatol=1e-2, fatol=1e-3,
             return_all_data=False):

    class RestartError(Exception):
        def __init__(self, param, *args, **kwargs):
            super().__init__(*args,**kwargs)
            self.param = param


    class NelderMeadError(Exception):
        pass

    def same_parameter(params, exps, *args, **kwargs):
        if len(params) > 4:
            bool_tmp = True
            for x in range(2,6):
                bool_tmp = bool_tmp and np.linalg.norm(params[-1]-params[-x])
            if bool_tmp:
                raise RestartError(params[-1])


    if ansatz_ is None:
        ansatz_ = ansatz.multi_particle

    # All options to Nelder-Mead
    disp_options = {'disp': display_after_run, 'xatol': xatol, 'fatol': fatol,
                    'maxiter': maxiter}

    vqe = vqeOverride.VQE_override(minimizer=minimize,
                                   minimizer_kwargs={'method':
                                                         opt_algorithm,
                                                     'options': disp_options})
    # If disp_run_info is True we will print every step of the Nelder-Mead


    for i in range(100):
        try:
            eig = vqe.vqe_run(ansatz_, H, initial_params, samples=samples, qc=qc,
                          disp=True, return_all=True, callback=same_parameter)
            if return_all_data:
                return eig
            else:
                eigval = eig['fun']
                optparam = eig['x']
                return eigval, optparam
        except RestartError as e:
            initial_params = e.param

    raise NelderMeadError('Fuck Nelder-Mead')

def smallest_dynamic(H, qc, initial_params,
                     ansatz_=None,
                     samples=1000,
                     opt_algorithm='Nelder-Mead',
                     maxiter=10000,
                     disp_run_info=True,
                     display_after_run=False,
                     xatol=1e-2, fatol=1e-3,
                     return_all_data=False,
                     convert_op=matrix_to_op.multi_particle,
                     print_option=None):
    if samples is None:
        raise TypeError(
            'Samples must be a number in this method. Use smallest() instead')
    disp_options = {'disp': display_after_run, 'xatol': xatol, 'fatol': fatol,
                    'maxiter': maxiter}
    vqe = vqeOverride.VQE_override(minimizer=minimize,
                                   minimizer_kwargs={'method':
                                                         opt_algorithm,
                                                     'options': disp_options})
    # Starting values of iteration:
    samp = 100
    params = initial_params
    # Do one meas to get initial variance
    eig_start, var_start = vqe.expectation(
        ansatz.multi_particle(params), H,
        samples=samp,
        qc=qc)
    # Tolerance is 2 std. deviations. Maybe change?
    tol = 2*np.sqrt(var_start)

    if tol < fatol:
        raise ValueError('fatol too large, already satisfied')
    while tol > fatol:
        run = vqe.vqe_run(ansatz_, H, params, samples=samp, qc=qc,
                          disp=disp_run_info, return_all=True)

        params = run['x']
        tol

        print('Increasing samples')
        params = result[1]
        samples *= 4
        tol *= 0.5

    return smallest(H, qc, params, samples=samp, fatol=tol, ansatz_=ansatz_,
                    xatol=xatol, return_all_data=return_all_data,
                    maxiter=maxiter,
                    disp_run_info=disp_run_info,
                    opt_algorithm=opt_algorithm,
                    display_after_run=display_after_run)


def negative(h, ansatz, qvm, num_eigvals=None,
             num_samples=None,
             opt_algorithm='L-BFGS-B',
             initial_params=None):
    """
    Calculates all negative or specified amount of eigs for a
    given hamiltonian matrix.

    @author: Eric

    :param h: np.array hamiltonian matrix
    :param ansatz: ansatz function
    :param num_eigvals: number of desired eigs to be calculated
    :param num_samples: number of samples on the qvm
    :param opt_algorithm:
    :param initial_params: ansatz parameters
    :return: list of energies
    """
    # TODO: Needs to be updated to fit new smallest
    if num_eigvals is None:
        num_eigvals = h.shape[0]
    energy = []
    for i in range(num_eigvals):
        eigval, eigvect = smallest(h, ansatz, qvm, num_samples,
                                   opt_algorithm, initial_params)
        if eigval >= 0:
            if num_eigvals != h.shape[0]:
                print('Warning: Unable to find '
                      'the specified amount of eigs')
            return energy
        else:
            energy.append(eigval)
            # Maybe eigvect should be normalized??
            h = h + 1.1 * np.abs(energy[i]) * np.outer(eigvect, eigvect)
            # move found eigenvalue to > 0.
    return energy


def all(H, ansatz, qvm, num_eigvals=None,
        num_samples=None, opt_algorithm='L-BFGS-B',
        initial_params=None):
    """
    Calculates all or specified amount of eigs for an Hamiltonian matrix
    TODO: Make so it handles sparse matrices? Currently finds
     double zero eigs
    TODO: Needs to be updated to fit new smallest

    @author: Eric

    :param H: np.array hamiltonian matrix
    :param ansatz: ansatz function
    :param num_eigvals: number of desired eigs to be calculated
    :param num_samples: number of samples on the qvm
    :param opt_algorithm:
    :param initial_params: ansatz parameters
    :return: list of energies
    """
    energy = negative(H, ansatz, qvm, num_eigvals,
                      num_samples, opt_algorithm,
                      initial_params)

    if num_eigvals is not None and len(energy) < num_eigvals:
        energy = energy + [-x for x in
                           negative(-1 * H, ansatz,
                                    qvm,
                                    num_eigvals - len(
                                        energy),
                                    num_samples,
                                    opt_algorithm,
                                    initial_params)]
    if len(energy) < H.shape[0]:
        energy = energy + [-x for x in
                           negative(-1 * H, ansatz,
                                    qvm, num_eigvals,
                                    num_samples,
                                    opt_algorithm,
                                    initial_params)]
        for i in range(len(energy), H.shape[0]): energy.append(0)

    return energy


def update_householder(H, ansatz, _, x):
    """
    Updates the Hamiltonian by block diagonalization using a Householder
    transform to reduce the dimensionality by one in each step. This function
    were made for numpy-arrays (ndarrays) rather than sparse (which is preferred
    in multi_particle). Npte: x should be normalized

    If we are going to use Householder transformations to reduce dimensionality
    I believe that ndarrays is the way to go, since householder-matrices are
    dense. Since this strategy requires matrix-multiplication it might be faster
    to add a x^H x to the Hamiltonian; at least if a x^H x can be turned into
    operators efficiently(currently I don't know anything better than mat2op_2).

    Note that if ||n|| is small there might be stabilty-issues.

    @author: Joel
    """
    if x.shape[0] > 1:
        # Find suitable basis-vector to reflect to (with householder)
        y = np.zeros(x.shape[0])
        idx = int(abs(x[0]) > 0.5)  # this makes sure that householder is stable
        y[idx] = 1

        # Create Householder-matrix T
        n = x - y
        n = n / np.linalg.norm(n)
        T = np.eye(n.shape[0]) - 2 * np.outer(n, n)

        # Calculate new hamiltonian
        T.dot(H.dot(T, out=H), out=H)
        H = np.delete(H, idx, axis=0)
        H = np.delete(H, idx, axis=1)
    return H, ansatz
