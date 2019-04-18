import numpy as np
from core import maps


def alternate(size):
    """
    Creates alternating parameters for ansätze, which is better for larger
    V's in the Lipkin-model; that is, the larger (positive) the off-diagonal
    values are, the closer to the minimum eigenvalue these parameters
    approximate.

    @author: Carl

    :param size: int representing size of Hamiltonian matrix
    :return: array representing the initial parameter for optimization
    :rtype: np.ndarray
    """
    return np.array([(-1) ** (i + 1) for i in range(size - 1)]) / np.sqrt(size)


def alternate_stereographic(h):
    """
    Creates parameters corresponding to a state proportional to (1, -1, 1,
    ...) for ansätze using stereographic projection, which is better for larger
    V's in the Lipkin-model; that is, the larger (positive) the off-diagonal
    values are, the closer to the minimum eigenvalue these parameters
    approximate.

    @author: Joel

    :param np.ndarray h: the Hamiltonian matrix
    :return: array representing the initial parameter for optimization
    :rtype: np.ndarray
    """
    state = np.array([(-1) ** i for i in range(h.shape[0])]) \
        / np.sqrt(h.shape[0])
    pole = int(np.argmax(np.diag(h)))
    return maps.sphere_to_plane(state, pole=pole)


def ones(size):
    """
    Creates an initial state for an ansätze.

    @author: Joel, Eric, Carl, Axel, Sebastian

    :param size:  int representing the size of hamiltonian matrix
    :return: array of ones of proper length
    :rtype: np.ndarray
    """
    return np.ones(size - 1) / np.sqrt(size)


def ucc(size):
    """
    Creates an inital state for an anstaz.

    @author: Carl

    :param size:  int representing the size of hamiltonian matrix
    :return: array of ones of proper length
    :rtype: np.ndarray
    """
    return np.array([(-1) ** (i) for i in range(size - 1)]) / np.sqrt(size)


def zeros(size):
    """
    Creates an inital state for an anstaz.

    @author: Carl

    :param size:  int representing the size of hamiltonian matrix
    :return: array of ones of proper length
    :rtype: np.ndarray
    """
    return np.zeros(size - 1) / np.sqrt(size)
