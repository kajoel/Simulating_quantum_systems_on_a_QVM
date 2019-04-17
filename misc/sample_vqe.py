from core import vqeOverride
from scipy.optimize import minimize


def normal_nelder_mead(xatol=1e-2, fatol=1e-2, return_all = False, maxiter =
10000):
    '''
    @author: Sebastian
    :param xatol:
    :param fatol:
    :param return_all:
    :param maxiter:
    :return:
    '''
    disp_options = {'disp': False, 'xatol': xatol,
                    'fatol': fatol,
                    'maxiter': maxiter, 'return_all': return_all}
    vqe = vqeOverride.VQE_override(minimizer=minimize,
                                   minimizer_kwargs={'method':
                                                         'Nelder-Mead',
                                                     'options': disp_options})
    return vqe