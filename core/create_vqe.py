from core import vqe_override
from scipy.optimize import minimize
from skopt import gp_minimize


def default_nelder_mead(xatol=1e-2, fatol=1e-2, return_all = False, maxiter =
10000):
    '''
    @author: Sebastian
    :param xatol:
    :param fatol:
    :param return_all:
    :param maxiter:
    :return:
    '''
    disp_options = {'disp': False, 'xatol': xatol, 'fatol': fatol,
                    'maxiter': maxiter, 'return_all': return_all}

    return vqe_override.VQE_override(minimizer=minimize,
                                   minimizer_kwargs={'method': 'Nelder-Mead',
                                                     'options': disp_options})


def default_bayes(acq_func = "gp_hedge",      
                  n_calls = 15,          
                  n_random_starts= 4,         
                  random_state = 123,
                  n_jobs = 1):
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
    
if __name__ == '__main__':
    default_bayes()
    


