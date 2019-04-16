import numpy as np
from skopt import gp_minimize

# Imports for VQE
from scipy.optimize import minimize
from core import ansatz
from core import init_params
from core import matrix_to_op
from core import vqe_override


def smallest(H, qc, initial_params, vqe,
             ansatz_=None,
             samples=None,
             disp_run_info=True):
    """
    TODO: Fix this documentation. Below is not up to date.

    Finds the smallest eigenvalue and corresponding -vector of H using VQE.

    @author: Eric

    :param H: PauliSum of hamiltonian
    :param qc: quantumcomputer object
    :param initial_params: ansatz parameters
    :param vqe: Quantum variational eigensolver object
    :param ansatz_: ansatz function
    :param samples: number of samples on the qc
    :param disp_run_info: displays run info from vqe_run


    :return: depending on return_all_data, either dict or only eigvect and param
                value
    """
    if ansatz_ is None:
        # Default
        ansatz_ = ansatz.multi_particle

    eig = vqe.vqe_run(ansatz_, H, initial_params, samples=samples, qc=qc,
                      disp=disp_run_info, return_all=True)

    eig['fun'], _ = vqe.expectation(ansatz_(eig['x']), H,
                                    samples=samples, qc=qc)
    return eig


def smallest_bayes(H, qc,
                   dimension,
                   vqe, ansatz_,
                   samples=None,
                   disp_run_info=True):
    """
    Finds the smallest eigenvalue using a Bayesian optimization algoritm.
    @author: Axel, Eric

    TODO: Go into VQEOverride and look at what you can return, because now
    we are not getting the data from the Bayesian Optimization returned, only
    the exp_val, parameter and variance.

    :param H: PauliSum of hamiltonian
    :param qc: either qc or qvm object, depending on version
    :param dimension: A list of tuples, with the intervals for the parameters
    :param ansatz_: ansatz function
    :param samples: Number of samples on the qc
    :param disp_run_info: Displays all data during the run. (It is ALOT)

    :return: list of energies or all data from all opttimization runs.
    """

    eig = vqe.vqe_run(ansatz_, H, dimension, samples=samples, qc=qc,
                      disp=disp_run_info, return_all=True)

    eig['fun'], _ = vqe.expectation(ansatz_(eig['x']), H,
                                    samples=samples, qc=qc)

    return eig


def smallest_restart(H, qc, initial_params, vqe,
                     ansatz_=None,
                     samples=None,
                     max_para=7,
                     max_iter=10,
                     tol_para=1e-4,
                     increase_samples=0,
                     maxiter=10000,
                     display_after_run=False,
                     disp = False,
                     disp_iter = False):
    """
    @author: Sebastian, Carl

    :param H:
    :param qc:
    :param initial_params:
    :param ansatz_:
    :param samples:
    :param opt_algorithm:
    :param maxiter:
    :param display_after_run:
    :param xatol:
    :param fatol:
    :param return_all_data:
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
                raise vqe_override.BreakError()

    if ansatz_ is None:
        ansatz_ = ansatz.multi_particle

    # If disp_run_info is True we will print every step of the Nelder-Mead

    fun_evals = []
    for i in range(max_iter):
        if disp_iter:
            print("\niter: {}".format(i))
            print("samples: {}".format(samples))
            # print("fatol: {}".format(fatol))

        # Have to make new vqe every time, else the callback gets duplicated
        # each iter (bug?)
        result = vqe.vqe_run(ansatz_, H, initial_params, samples=samples,
                             qc=qc,
                             disp=disp, return_all=True,
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


def negative(h, qc, ansatz_, vqe, parameters, samples,
             topauli_method=matrix_to_op.multi_particle,
             num_eigvals=None, bayes=True, disp_run_info=False):
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
    :param num_eigvals: amount of eigenvalues to be found
    :param bayes: True for baysian optimization, false for Nelder-Mead
    :param disp_run_info: do you want to kill your terminal?
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
                                 disp_run_info=disp_run_info)

        else:
            eig = smallest(H, qc, ansatz_=ansatz_, vqe=vqe, samples=samples,
                           initial_params=parameters,
                           disp_run_info=disp_run_info)
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

            h = h + 1.1 * np.abs(eig['fun']) * np.outer(eigvect, eigvect)
            print(h)
            # move found eigenvalue to > 0.
            print('Found eigenvalue at:', eig['fun'])
    return energy


def all(h, qc, ansatz_, vqe, parameters, samples,
        topauli_method=matrix_to_op.multi_particle,
        num_eigvals=None, bayes=True,
        disp_run_info=False):
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
    :param disp_run_info: do you want to kill your terminal?
    :return: list of dicts of results
    """
    energy = negative(h, qc, ansatz_, vqe, parameters, samples, topauli_method,
                      num_eigvals, bayes, disp_run_info)
    newen = []
    if num_eigvals is not None and len(energy) < num_eigvals:
        newen = negative(-h, qc, ansatz_, vqe, parameters, samples,
                         topauli_method,
                         num_eigvals - len(energy), bayes,
                         disp_run_info)
    if len(energy) < h.shape[0]:
        print('switching eigs')
        newen = negative(-h, qc, ansatz_, vqe, parameters, samples,
                         topauli_method,
                         num_eigvals, bayes,
                         disp_run_info)
    for i in range(len(newen)):
        newen[i]['fun'] = -1 * newen[i]['fun']
    energy += newen
    return energy
