import numpy as np
from core import ansatz


def _test_depth(ansatz, n_min=1, n_max=12, m=5):
    nn = n_max - n_min + 1
    nbr_ops = np.zeros(nn)
    for i in range(nn):
        n = n_min + i
        temp = np.empty(m)
        for j in range(m):
            temp[j] = len(ansatz(n+1)(np.random.randn(n)))
        nbr_ops[i] = np.average(temp)
        print(n)
    return nbr_ops


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    ansatz1 = ansatz.one_particle_ucc
    ansatz2 = ansatz.one_particle

    n_max1 = 10
    n_max2 = 10

    nbr_ops_1 = _test_depth(ansatz1, n_max=n_max1)
    nbr_ops_2 = _test_depth(ansatz2, n_max=n_max2)

    if n_max1 == n_max2:
        plt.figure(0)

        plt.plot(nbr_ops_1)
        plt.plot(nbr_ops_2)

        plt.legend(["Number of gates in {}".format(ansatz1.__name__),
                    "Number of gates in {}".format(ansatz2.__name__)])

        plt.show()
    else:
        plt.figure(0)
        plt.plot(nbr_ops_1)
        plt.title("Number of gates in {}".format(ansatz1.__name__))

        plt.figure(1)
        plt.plot(nbr_ops_2)
        plt.title("Number of gates in {}".format(ansatz2.__name__))
        plt.show()
