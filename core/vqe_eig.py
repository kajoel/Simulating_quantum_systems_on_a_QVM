
def smallest(H, qc, initial_params, vqe,
             ansatz_,

    :return: depending on return_all_data, either dict or only eigvect and param
                value
    """
    if ansatz_ is None:
        # Default
        ansatz_ = ansatz.multi_particle

    eig = vqe.vqe_run(ansatz_, H, initial_params, samples=samples, qc=qc,
                      disp=disp_run_info, return_all=True)
def negative(h, qc, ansatz_, vqe, parameters, samples,
             topauli_method=matrix_to_op.multi_particle,
             num_eigvals=None, bayes=True, disp_run_info=False):
    """
    if bayes and not isinstance(parameters[0], tuple):
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
