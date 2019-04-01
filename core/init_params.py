import numpy as np


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


def ones(size):
    """
    Creates an inital state for an anstaz.

    @author: Joel, Eric, Carl, Axel

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