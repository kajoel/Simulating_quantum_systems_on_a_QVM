"""
Created on Mon Mar  4 10:50:49 2019
"""

# Imports
import numpy as np
from pyquil.quil import Program
from grove.alpha.arbitrary_state.arbitrary_state import create_arbitrary_state


def one_particle(theta: np.ndarray) -> Program:
    """
    @author: Joel, Carl
    Creates a program to set up an arbitrary one-particle-state.
    :param theta: Vector representing the state.
    :return: PyQuil program setting up the state.
    """

    vector = np.zeros(2 ** (theta.shape[0] + 1))
    vector[1] = 1
    vector[[2 ** (i + 1) for i in range(theta.shape[0])]] = theta
    vector *= 1 / np.linalg.norm(vector)
    return create_arbitrary_state(vector)


def multi_particle(theta: np.ndarray) -> Program:
    """
    @author: Joel
    Creates a program to set up an arbitrary state.
    :param theta: Vector representing the state.
    :return: PyQuil program setting up the state.
    """
    theta = np.concatenate((np.array([1])/(theta.size + 1), theta), axis=0)
    return create_arbitrary_state(theta)
