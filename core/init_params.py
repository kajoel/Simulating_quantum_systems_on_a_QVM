import numpy as np


def one_particle_ones(size):
    """
    @author: Carl, Axel
    Creates the initial state for the one_partical_ansatz
    :param size: int representing size of Hamiltonian matrix
    :return: array representing the initial parameter for optimization
    :rtype: np.ndarray
    """
    return 1 / np.sqrt(size) * np.array([1 for i in range(size - 1)])


def one_particle_alt(size):
    """
    @author: Carl
    Creates the best initial state for the one_particle
    :param size: int representing size of Hamiltonian matrix
    :return: array representing the initial parameter for optimization
    :rtype: np.ndarray
    """
    return 1 / np.sqrt(size) * np.array(
        [(-1) ** (i + 1) for i in range(size - 1)])


def multi_particle_ones(size):
    """
    @author: Joel, Eric
    Creates an inital state for the multi_particle anstaz
    :param size:  int representing the size of hamiltonian matrix
    :return: array of ones of proper length
    :rtype: np.ndarray
    """
    return np.ones(size-1) / np.sqrt(size)
