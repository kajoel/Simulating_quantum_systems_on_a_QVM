"""
File with callback-methods for e.g. dynamical stopping and live plotting.
It's recommended that all callbacks support both *args and **kwargs for
future compatibility.
"""
import numpy as np
import matplotlib.pyplot as plt
from core.vqe_override import BreakError


def merge_callbacks(*args):
    """
    If you want to use merge_callbacks callbacks, use merge_callbacks(callback_1,
    callback_2, ...).

    @author = Joel

    :param args: callbacks to use
    :return: callback function which can be passed to VQE.
    """

    def callback(*x, **y):
        for arg in args:
            arg(*x, **y)

    return callback


def scatter(**kwargs):
    def callback(x, y, *_, **__):
        plt.clf()
        plt.scatter(x, y, **kwargs)
        plt.pause(0.05)

    return callback


def stop_dynamically(xatol, fatol):
    """
    Callback for dynamical stopping.
    TODO

    @author = Joel

    :param xatol:
    :param fatol:
    :return:
    """

    def callback(params, exps, *_, **__):
        pass

    return callback


def stop_if_stuck():
    '''

    @author = Sebastian
    :return:
    '''

    def callback(params, *_, **__):
        if len(params) > 1 and params[-1][0] == params[-2][0]:  # Notera och
            # uppskatta att denna if sats inte kan skapa error då andra
            # uttrycket inte evalueras om det första är falskt, ytterst
            # tillfredställande.
            raise BreakError

    return callback


def is_on_same_parameter():
    '''

    @author = Sebastian

    Demonstrates that Nelder-Mead does get stuck on the exact same parameter
    :return:
    '''

    def callback(params, *_, **__):
        if len(params) > 1:
            print("Parameter the same as previous iter: {}".format(
                params[-1][0] == params[-2][0]))

    return callback


def stop_if_stuck_x_times(x, tol=0):
    '''

    @author = Sebastian

    :param x:
    :return:
    '''

    def callback(params, *args, **kwargs):
        if len(params) >= x:
            bool_tmp = True
            for i in range(2, x + 1):
                bool_tmp = bool_tmp and np.linalg.norm(params[-1][0] -
                                                       params[-i][0]) <= tol
            if bool_tmp:
                # raise RestartError(params[-1])
                raise BreakError

    return callback
