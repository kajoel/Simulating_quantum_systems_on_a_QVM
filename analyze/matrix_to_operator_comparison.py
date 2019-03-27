"""
Comparison of different methods from matrix_to_operator
"""
import numpy as np
from core import matrix_to_op


def _test_mat_to_op(hamiltonian_operator, jmin=0.5, jmax=100, tol=1e-8):
    """
    Tests that eigs computed using a function that generates hamiltonian
    operators are correct. This test doesn't check for extra eigs but
    only that the ones that should be there is there.
    @author: kajoel
    """
    from lipkin_quasi_spin import hamiltonian, eigs
    from openfermion.transforms import get_sparse_operator

    no_error = True
    for j2 in range(round(2 * jmin), round(2 * jmax) + 1):
        j = j2 / 2
        print("j = " + str(j))
        V = float(np.random.randn(1))
        H = hamiltonian(j, V)
        E = eigs(j, V)
        for i in range(len(H)):
            H_op = hamiltonian_operator(H[i])
            H_op = get_sparse_operator(H_op).toarray()
            E_op = np.linalg.eigvals(H_op)
            # Check that E_op contains all eigs in E[i]
            for E_ in E[i]:
                if all(abs(E_op - E_) > tol):
                    no_error = False
                    print("Max diff: " + str(max(abs(E_op - E_))))

    if no_error:
        print("Success!")
    else:
        print("Fail!")


def _test_complexity(mat2op, jmin=0.5, jmax=25):
    from core import lipkin_quasi_spin

    n = 1 + round(2 * (jmax - jmin))
    nbr_terms = np.empty(2 * n)
    max_nbr_ops = np.zeros(2 * n)
    matrix_size = np.empty(2 * n)
    for i in range(n):
        j = jmin + 0.5 * i
        print(j)
        H = lipkin_quasi_spin.hamiltonian(j, np.random.randn(1)[0])
        for k in range(len(H)):
            matrix_size[2 * i + 1 - k] = H[k].shape[0]
            terms = mat2op(H[k])
            nbr_terms[2 * i + 1 - k] = len(terms)
            for term in terms:
                if len(term) > max_nbr_ops[2 * i + 1 - k]:
                    max_nbr_ops[2 * i + 1 - k] = len(term)

    return matrix_size, nbr_terms, max_nbr_ops


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    matrix_size_1, nbr_terms_1, max_nbr_ops_1 = \
        _test_complexity(matrix_to_op.one_particle)
    matrix_size_2, nbr_terms_2, max_nbr_ops_2 = \
        _test_complexity(matrix_to_op.multi_particle)

    if not all(matrix_size_1 == matrix_size_2):
        raise Exception("Something went wrong with the sizes.")

    plt.figure(0)
    plt.plot(matrix_size_1, np.array([nbr_terms_1, nbr_terms_2]).T)
    plt.title("Number of Pauli terms")
    plt.legend(["a^dag a", "qubit op"])

    plt.figure(1)
    plt.plot(matrix_size_1, np.array([max_nbr_ops_1, max_nbr_ops_2]).T)
    plt.title("Maximum number of ops per term")
    plt.legend(["a^dag a", "qubit op"])
    plt.show()

    #  test_mat_to_op(one_particle, jmax=10)
    # The following tests has (successfully) been completed:
    # _test_mat_to_op(one_particle)
    # _test_mat_to_op(one_particle, jmax=11)

# get_sparse_operator seems to permute the basis. For 2 qubits:
# the permutation is (0 2 1 3) and for 3 qubits it's (0 4 2 6 1 5 3 7)
# or, in binary, (00, 10, 01, 11) and (000, 100, 010, 110, 001, 101, 011, 111).
