import numpy as np


def alternate(size):
    """
    @author: Carl
    Creates alternating parameters for ansätze, which is better for larger
    V's in the Lipkin-model; that is, the larger (positive) the off-diagonal
    values are, the closer to the minimum eigenvalue these parameters
    approximate
    :param size: int representing size of Hamiltonian matrix
    :return: array representing the initial parameter for optimization
    :rtype: np.ndarray
    """
    return 1 / np.sqrt(size) * np.array(
        [(-1) ** (i + 1) for i in range(size - 1)])


def ones(size):
    """
    @author: Joel, Eric, Carl, Axel
    Creates an inital state for an anstaz.
    :param size:  int representing the size of hamiltonian matrix
    :return: array of ones of proper length
    :rtype: np.ndarray
    """
    return np.ones(size - 1) / np.sqrt(size)
