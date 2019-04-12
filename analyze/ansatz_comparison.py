import numpy as np
from core import ansatz
from core import data


def compare_and_save():
    ansatzs = [ansatz.one_particle, ansatz.one_particle_ucc,
               ansatz.multi_particle, ansatz.multi_particle_ucc]

    metadata = {'description': '"gates" is a list of numpy arrays. Each array '
                               'contains the number of gates in a typical '
                               'ansatz-program for different matrix-sizes. '
                               'See "ansatz" in metadata for the ansätze and '
                               '"mat_size" in data for matrix-sizes.',
                'ansatz': [ansatz_.__name__ for ansatz_ in ansatzs]}

    file = 'mat_to_op and ansatz/ansatz_depth'

    n_max = 100
    max_ops = 10000

    gates = []
    mat_size = []

    data_ = {'gates': gates, 'mat_size': mat_size}

    for ansatz_ in ansatzs:
        nbr_ops = _test_depth(ansatz_, n_max=n_max, max_ops=max_ops)
        gates.append(nbr_ops)
        mat_size.append(np.linspace(2, 1+nbr_ops.size, nbr_ops.size,
                                    dtype=np.uint16))

    data.save(file=file, data=data_, metadata=metadata)


def _test_depth(ansatz, n_min=1, n_max=12, m=5, max_ops=10000):
    nn = n_max - n_min + 1
    nbr_ops = np.zeros(nn)
    for i in range(nn):
        n = n_min + i
        temp = np.empty(m)
        for j in range(m):
            temp[j] = len(ansatz(n+1)(np.random.randn(n)))
        nbr_ops[i] = np.average(temp)
        print(n)
        if nbr_ops[i] > max_ops:
            return nbr_ops[:i+1]
    return nbr_ops


def compare_and_plot():
    import matplotlib.pyplot as plt

    ansatz1 = ansatz.one_particle
    ansatz2 = ansatz.one_particle_ucc

    n_max1 = 15
    n_max2 = 0

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


def plot_from_save():
    pass


if __name__ == "__main__":
    compare_and_save()
