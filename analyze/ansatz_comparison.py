import numpy as np
from core import ansatz


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


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    nbr_ops_s = _test_depth(ansatz.one_particle)
    nbr_ops_m = _test_depth(ansatz.multi_particle, n_max=2 ** 5)

    plt.figure(0)
    plt.plot(nbr_ops_s)
    plt.title("Number of gates in one-particle-ansatz")

    plt.figure(1)
    plt.plot(nbr_ops_m)
    plt.title("Number of gates in multi-particle-ansatz")
