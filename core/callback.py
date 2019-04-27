"""
File with callback-methods for e.g. dynamical stopping and live plotting.
It's recommended that all callbacks support both *args and **kwargs for
future compatibility.
"""
import numpy as np
import matplotlib.pyplot as plt

from core import vqe_override
from core.vqe_override import BreakError, RestartError


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


def restart_break(max_same_para, tol_para, disp=False):
    """
    Callback (new) for restarting and breaking.

    @author = Joel, Carl

    :param max_same_para: how many times to stand still before restarting
    :param tol_para: how close is considered to stand "still"
    :param disp: if True, print Restarting when restarting
    :return: callback function which can be passed to VQE.
    """
    last_restart = None
    restarts = 0

    def same_parameter(params, *args, **kwargs):
        nonlocal last_restart, restarts
        if len(params) > max_same_para - 1:
            bool_tmp = True
            for x in range(2, max_same_para + 1):
                bool_tmp = bool_tmp and np.linalg.norm(params[-1] - params[-x])\
                           < tol_para

            if bool_tmp:
                restarts += 1
                if last_restart is not None and \
                        np.linalg.norm(params[-1] - last_restart) < tol_para:
                    raise BreakError
                else:
                    last_restart = params[-1]
                    if disp:
                        print(f"\nRestart: {restarts}")
                    raise RestartError

    return same_parameter


def restart(max_same_para, tol_para, disp = False):
    """
    @author: Sebastian, Carl

    :param max_same_para: how many times to stand still before restarting
    :param tol_para: how close is considered to stand "still"
    :param disp: if True, print Restarting when restarting
    :return: callback function which can be passed to VQE.
    """

    def same_parameter(params, *args, **kwargs):
        if len(params) > max_same_para - 1:
            bool_tmp = True
            for x in range(2, max_same_para + 1):
                bool_tmp = bool_tmp and np.linalg.norm(params[-1] - params[-x])\
                           < tol_para
            if bool_tmp:
                if disp:
                    print("Restarting")
                raise RestartError

    return same_parameter


def scatter(**kwargs):
    '''
    @author: Sebastian
    :param kwargs:
    :return: callback function which can be passed to VQE.
    '''

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
    :return: callback function which can be passed to VQE.
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
    :return: callback function which can be passed to VQE.
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
    :return: callback function which can be passed to VQE.
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


def trigger_only_every_x_iter(x, callback):
    '''

    @author: Sebastian

    :param x: nums of iters to wait before calling callback
    :param callback:
    :return: callback function which can be passed to VQE.
    '''
    n = 0

    def callback2(*args, **kwargs):
        nonlocal n
        n = (n + 1) % x
        if n == 0:
            callback(*args, **kwargs)

    return callback2
