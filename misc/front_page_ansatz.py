"""
Module with special ansatz for the front page.
"""
import numpy as np
from scipy import sparse
from grove.alpha.arbitrary_state.arbitrary_state import create_arbitrary_state
from core import maps


def ansatz(h: sparse.spmatrix):
    eigs = np.linalg.eigvalsh(h.todense())
    idx_min = int(np.argmin(eigs))
    idx_max = int(np.argmax(eigs))
    idx_mid = int(np.argmin(np.abs(eigs - 0.5*(eigs[idx_min] + eigs[idx_max]))))

    def wrap(theta: np.ndarray):
        """
        Creates arbitrary state.

        :param theta: Vector representing the state.
        :return: PyQuil program setting up the state.
        """
        return create_arbitrary_state(plane_to_hemisphere(theta,
                                                          idx_min, idx_mid))

    return wrap


def plane_to_hemisphere(x: np.ndarray, pole_origin: int, pole_inf: int):
    # Map vector in plane to sphere
    y = maps.plane_to_sphere(x, pole_inf)
    # Halve angle to pole_origin
    r = np.zeros(y.shape)
    r[pole_origin] = 1
    theta = np.arccos(np.dot(y, r)/np.linalg.norm(y))
    y[pole_origin] *= np.cos(theta/2)/np.cos(theta)
    if np.sin(theta) != 0:
        y[np.arange(y.size) != pole_origin] *= np.sin(theta/2)/np.sin(theta)
    # Rotate
    return x
