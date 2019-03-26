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
    # TODO: doc behaves weird, theta: Union[ndarray, ndarray]

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
    theta = np.concatenate((np.array([1]), theta), axis=0)
    return create_arbitrary_state(theta)


################################################################################
# TESTS
################################################################################
def _test_depth(ansatz, n_min=1, n_max=12, m=5):
    nn = n_max - n_min + 1
    nbr_ops = np.zeros(nn)
    for i in range(nn):
        n = n_min + i
        print(n)
        temp = np.empty(m)
        for j in range(m):
            temp[j] = len(ansatz(np.random.randn(n)))
        nbr_ops[i] = np.average(temp)
    return nbr_ops


################################################################################
# MAIN
################################################################################
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    nbr_ops_s = _test_depth(one_particle)
    nbr_ops_m = _test_depth(multi_particle, n_max=2 ** 5)

    plt.figure(0)
    plt.plot(nbr_ops_s)
    plt.title("Number of gates in one-particle-ansatz")

    plt.figure(1)
    plt.plot(nbr_ops_m)
    plt.title("Number of gates in multi-particle-ansatz")
