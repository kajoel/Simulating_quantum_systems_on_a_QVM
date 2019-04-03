"""
File with callback-methods for e.g. dynamical stopping and live plotting.
It's recommended that all callbacks support both *args and **kwargs for
future compatibility.
"""


def multiple(*args):
    """
    If you want to use multiple callbacks, use multiple(callback_1,
    callback_2, ...).

    @author = Joel

    :param args: callbacks to use
    :return: callback function which can be passed to VQE.
    """
    def callback(*x, **_):
        for arg in args:
            arg(*x)

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
