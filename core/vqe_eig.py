"""
Created on Wed Mar  6 16:35:25 2019
"""

import numpy as np
from skopt import gp_minimize


# Imports for VQE
from scipy.optimize import minimize
from core import ansatz, vqeOverride
from core import init_params
from core import matrix_to_op


def smallest(H, qc, initial_params, vqe,
             ansatz_=None,
             samples=None,
             maxiter=10000,
             disp_run_info=True,
             display_after_run=False,
             xatol=1e-2, fatol=1e-3,
             return_all_data=False, callback=None, vqe = None):
    """
    TODO: Fix this documentation. Below is not up to date.

    Finds the smallest eigenvalue and corresponding -vector of H using VQE.

    @author: Eric, Axel, Carl

    :param H: PauliSum of hamiltonian
    :param qc: quantumcomputer object
    :param initial_params: ansatz parameters
    :param vqe: Quantum variational eigensolver object
    :param ansatz_: ansatz function
    :param samples: number of samples on the qc
    :param maxiter: maximum number of iterations
    :param disp_run_info: displays run info from vqe_run
    :param display_after_run: show Nelder-Mead algorithm run info
    :param xatol: parameter tolerance
    :param fatol: function tolerance
    :param return_all_data: If true, returns dict of all run data (i.e.
                        variance). Else returns eigvect and parameter value

    :return: depending on return_all_data, either dict or only eigvect and param
                value
    """

    if ansatz_ is None:
        # Default
        ansatz_ = ansatz.multi_particle

    # All options to Nelder-Mead
    if vqe == None:
        disp_options = {'disp': display_after_run, 'xatol': xatol, 'fatol': fatol,
                        'maxiter': maxiter}

    # vqe = vqeOverride.VQE_override(minimizer=minimize,
    #                               minimizer_kwargs={'method':
    #                                                     'Nelder-Mead',
    #                                                 'options': disp_options})
        vqe = vqeOverride.VQE_override(minimizer=minimize,
                                       minimizer_kwargs={'method':
                                                             opt_algorithm,
                                                         'options': disp_options})

    # If disp_run_info is True we will print every step of the Nelder-Mead

    eig = vqe.vqe_run(ansatz_, H, initial_params, samples=samples, qc=qc,
                      disp=disp_run_info, return_all=True)

    return eig


def smallest_restart(H, qc, initial_params,
                     ansatz_=None,
                     samples=None,
                     max_para=5,
                     max_iter=10,
                     tol_para=1e-4,
                     increase_samples=0,
                     disp_iter=False, vqe=None):
    """
    @author: Sebastian, Carl

    :param H:
    :param qc:
    :param initial_params:
    :param ansatz_:
    :param samples:
    :param max_para:
    :param max_iter:
    :param tol_para:
    :param increase_samples:
    :param disp_iter:
    :param vqe:
    :return:
    """

    def same_parameter(params, *args, **kwargs):
        if len(params) > max_para - 1:
            bool_tmp = True
            for x in range(2, max_para + 1):
                bool_tmp = bool_tmp and np.linalg.norm(params[-1] - params[-x]) \
                           < tol_para
            if bool_tmp:
                # raise RestartError(params[-1])
                raise vqeOverride.BreakError()

    if ansatz_ is None:
        ansatz_ = ansatz.multi_particle

    # If disp_run_info is True we will print every step of the Nelder-Mead

    fun_evals = []
    for i in range(max_iter):
        if disp_iter:
            print("\niter: {}".format(i))
            print("samples: {}".format(samples))

        # Have to make new vqe every time, else the callback gets duplicated
        # each iter (bug?)
        result = vqe.vqe_run(ansatz_, H, initial_params, samples=samples,
                             qc=qc,
                             disp=True, return_all=True,
                             callback=same_parameter)
        params = result['iteration_params']
        fun_evals.append(result['fun_evals'])
        if len(params) > max_para - 1:
            bool_tmp = True
            for x in range(2, max_para + 1):
                bool_tmp = bool_tmp and np.linalg.norm(params[-1] - params[-x]) \
                           < tol_para
            if bool_tmp:
                initial_params = params[-1]
                samples += increase_samples
            else:
                result['fun_evals'] = fun_evals
                return result

    print('Did not finish after {} iterations'.format(max_iter))
    result['fun_evals'] = fun_evals
    return result


class RestartError(Exception):
    def __init__(self, param, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.param = param


class NelderMeadError(Exception):
    pass


def smallest_bayes(H, qc,
                   dimension,
                   ansatz_,
                   samples=None,
                   disp=True,
                   acq_func="gp_hedge",
                   n_calls=30,
                   n_random_starts=5,
                   random_state=123,
                   x0=None):
                   samples=None,
                   return_all_data=False,
                   disp=True,
                   acq_func="gp_hedge",
                   n_calls=30,
                   n_random_starts=5,
                   random_state=123,
                   x0=None):
    """
    Finds the smallest eigenvalue using a Bayesian optimization algoritm.     
    @author: Axel
    
    TODO: Go into VQEOverride and look at what you can return, because now
    we are not getting the data from the Bayesian Optimization returned, only 
    the exp_val, parameter and variance. 

    :param H: PauliSum of hamiltonian
    :param qc: either qc or qvm object, depending on version
    :param dimension: A list of tuples, with the intervals for the parameters 
    :param ansatz_: ansatz function
    :param samples: Number of samples on the qvm
    :param return_all_data: If True returns data from all the runs.
    :param disp: Displays all data during the run. (It is ALOT)
    
    For the parameters below, see the skopt documentation: 
    https://github.com/scikit-optimize/scikit-optimize
    :param acq_func: Function to minimize over the gaussian prior. 
    :param n_calls: Number of calls to `func`
    :param n_random_starts: Number of evaluations of `func` with random points 
                            before approximating it with `base_estimator`.
    :random_state: Set random state to something other than None for 
                   reproducible results.
    :return: list of energies or all data from all opttimization runs.
    """

    # All options to Bayes opt
    opt_options = {'acq_func': acq_func,
                   'n_calls': n_calls,
                   'n_random_starts': n_random_starts,
                   'random_state': random_state}

    vqe = vqeOverride.VQE_override(minimizer=gp_minimize,
                                   minimizer_kwargs=opt_options)

    # Run to calculate the noise level
    initial_param = [param[0] for param in dimension]
    _, noise = vqe.expectation(ansatz_(initial_param), H,
                               samples=samples, qc=qc)

    opt_options['noise'] = noise

    # Need to initiate the vqe again so we can give it the variance as noise
    vqe = vqeOverride.VQE_override(minimizer=gp_minimize,
                                   minimizer_kwargs=opt_options)

    # The actual run
    eig = vqe.vqe_run(ansatz_, H, dimension, samples=samples, qc=qc,
                      disp=disp, return_all=True)

    eig['fun'], _ = vqe.expectation(ansatz_(eig['x']), H,
                                    samples=samples, qc=qc)

    eig['expectation_vars'] = noise

    return eig


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
    tol = 2 * np.sqrt(var_start)

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

    return smallest(H, qc, params, vqe, samples=samp, fatol=tol,
                    ansatz_=ansatz_,
                    xatol=xatol, return_all_data=return_all_data,
                    maxiter=maxiter,
                    disp_run_info=disp_run_info,
                    display_after_run=display_after_run)


def negative(h, qc, ansatz_, vqe, parameters, samples,
             topauli_method=matrix_to_op.multi_particle,
             num_eigvals=None, bayes=True,
             maxiter=10000, disp_run_info=False, xatol=1e-2, fatol=1e-3,
             n_calls=30,
             n_random_starts=5, ):
    """
    Calculates all (or specified amount of) negative eigenvalues using an
    emulated quantum computer.
    @author: Eric
    :param h: numpy ndarray of hamiltonian
    :param qc: qunatum computer object
    :param ansatz_: ansatz function
    :param vqe: Variational Quantum Eigensolver object
    :param parameters: parameters to ansatz OR params to sweep over for bayesian
    :param samples: total number of samples, or None
    :param topauli_method: method to convert matrix to paulisum
    :param num_eigvals: amout of eigenvalues to be found
    :param bayes: True for beysian optimization, false for Nelder-Mead
    :param maxiter: max number of iteratiosn
    :param disp_run_info: do you want to kill your terminal?
    :param xatol: x tol
    :param fatol: funtol
    :param n_calls: number of function calls for bayesian algoritm
    :param n_random_starts: number of random starts in bayesian algoritm
    :return: list of dicts of results

    """
    if bayes and not isinstance(parameters[0], tuple):
        raise TypeError(
            'parameters must be a list of tuples for Bayesian optimization')
    elif not bayes and not isinstance(parameters, np.ndarray):
        raise TypeError(
            'parameters must be an ndarray for Classical Nelder-Mead '
            'optimization')

    if num_eigvals is None:
        num_eigvals = h.shape[0]
    energy = []

    for i in range(num_eigvals):
        H = topauli_method(h)
        if bayes:
            eig = smallest_bayes(H, qc,
                                 dimension=parameters,
                                 ansatz_=ansatz_,
                                 samples=samples,
                                 disp=disp_run_info, n_calls=n_calls,
                                 n_random_starts=n_random_starts)

        else:
            eig = smallest(H, qc, ansatz_=ansatz_, vqe=vqe, samples=samples,
                           initial_params=parameters, maxiter=maxiter,
                           disp_run_info=disp_run_info, xatol=xatol,
                           fatol=fatol)
        eig['fun'] = vqe.expectation(
            ansatz_(eig['x']),
            H,
            samples,
            qc)

        if eig['fun'] >= 0:
            if num_eigvals != h.shape[0]:
                print('Warning: Unable to find '
                      'the specified amount of eigs')
            return energy
        else:
            energy.append(eig)
            # Maybe eigvect should be normalized??

            eigvect = np.concatenate(
                (np.array([1 / np.sqrt(h.shape[0])]), np.asarray(eig['x'])),
                axis=0)
            eigvect = np.asarray(eig['x'])
            eigvect = eigvect / np.linalg.norm(eigvect)

            h = h + 2.0 * np.abs(eig['fun']) * np.outer(eigvect, eigvect)
            print(h)
            # move found eigenvalue to > 0.
            print('Found eigenvalue at:', eig['fun'])
    return energy


def all(h, qc, ansatz_, vqe, parameters, samples,
        topauli_method=matrix_to_op.multi_particle,
        num_eigvals=None, bayes=True,
        maxiter=10000, disp_run_info=False, xatol=1e-2, fatol=1e-3, n_calls=30,
        n_random_starts=5):
    """
    Finds all eigenvalues for a given hamiltonian matrix
    @author: Eric
    :param h: numpy ndarray of hamiltonian
    :param qc: qunatum computer object
    :param ansatz_: ansatz function
    :param vqe: Variational Quantum Eigensolver object
    :param parameters: parameters to ansatz OR params to sweep over for bayesian
    :param samples: total number of samples, or None
    :param topauli_method: method to convert matrix to paulisum
    :param num_eigvals: amout of eigenvalues to be found
    :param bayes: True for beysian optimization, false for Nelder-Mead
    :param maxiter: max number of iteratiosn
    :param disp_run_info: do you want to kill your terminal?
    :param xatol: x tol
    :param fatol: funtol
    :param n_calls: number of function calls for bayesian algoritm
    :param n_random_starts: number of random starts in bayesian algoritm
    :return: list of dicts of results
    """
    energy = negative(h, qc, ansatz_, vqe, parameters, samples, topauli_method,
                      num_eigvals, bayes, maxiter, disp_run_info, xatol, fatol,
                      n_calls, n_random_starts)
    newen = []
    if num_eigvals is not None and len(energy) < num_eigvals:
        newen = negative(-h, qc, ansatz_, vqe, parameters, samples,
                         topauli_method,
                         num_eigvals - len(energy), bayes, maxiter,
                         disp_run_info,
                         xatol, fatol, n_calls, n_random_starts)
    if len(energy) < h.shape[0]:
        print('switching eigs')
        newen = negative(-h, qc, ansatz_, vqe, parameters, samples,
                         topauli_method,
                         num_eigvals, bayes, maxiter,
                         disp_run_info,
                         xatol, fatol, n_calls, n_random_starts)
    for i in range(len(newen)):
        newen[i]['fun'] = -1 * newen[i]['fun']
    energy += newen
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
