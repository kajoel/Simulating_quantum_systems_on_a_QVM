import numpy as np
# Imports for VQE
from core import ansatz
from core import matrix_to_op
from core import vqe_override


def smallest(H, qc, initial_params, vqe,
             ansatz_,
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
    TODO: Fix this documentation. Below is not up to date.

    Finds the smallest eigenvalue and corresponding -vector of H using VQE.

    @author: Eric, Axel, Carl, Sebastian

    :param H: PauliSum of hamiltonian
    :param qc: either qc or qvm object, depending on version
    :param ansatz_: ansatz function
    :param num_samples: number of samples on the qvm
    :param opt_algorithm:
    :param initial_params: ansatz parameters
    :return: list of energies
    """

    
    # All options to Nelder-Mead
    disp_options = {'disp': display_after_run, 'xatol': xatol, 'fatol': fatol,
                    'maxiter': maxiter}

    vqe = vqeOverride.VQE_override(minimizer=minimize,
                                   minimizer_kwargs={'method':
                                                         opt_algorithm,
                                                     'options': disp_options})
    # If disp_run_info is True we will print every step of the Nelder-Mead

    # print('Initial parameter:', initial_params, '\n')
    eig = vqe.vqe_run(ansatz_, H, initial_params, samples=samples, qc=qc,
                      disp=disp, return_all=return_all_data, callback=callback)

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
                     max_para=5,
                     max_iter=10,
                     tol_para=1e-4,
                     increase_samples=0,
                     opt_algorithm='Nelder-Mead',
                     maxiter=10000,
                     display_after_run=False,
                     disp = False,
                     disp_iter = False,
                     xatol=1e-2, fatol=1e-3):
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
                raise vqeOverride.BreakError()

    if ansatz_ is None:
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
                     ansatz_,
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
