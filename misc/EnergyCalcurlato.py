# Imports
# Openfermion:
from openfermion.transforms import jordan_wigner
from openfermion.ops import FermionOperator
# Forestopenfermion:
from forestopenfermion import qubitop_to_pyquilpauli
# PyQuil:
from pyquil.paulis import *
from pyquil.api import QVMConnection
# Scipy:
import scipy.sparse as sparse
from scipy.linalg import eigh
from scipy.optimize import minimize
# Numpy:
import numpy as np
# Grove:
from grove.pyvqe.vqe import VQE
from grove.alpha.arbitrary_state.arbitrary_state import create_arbitrary_state
# Other:
from lipkin_quasi_spin import hamiltonian, eigenvalues


# OriginalHamiltonian = np.array([[-5 / 2, np.sqrt(10), 0], [np.sqrt(10), -1 / 2, np.sqrt(18)], [0, np.sqrt(18), 3 / 2]])


# Joel, tweakad av Eric
def update_householder(H, x):
    if x.shape[0] > 1:
        # Find suitable basis-vector to reflect to (with householder)
        y = np.zeros(x.shape[0])
        idx = int(abs(x[0]) > 0.5)  # this makes sure that householder is stable
        y[idx] = 1

        # Create Householder-matrix T
        n = x-y
        n = n/np.linalg.norm(n)
        T = np.eye(n.shape[0]) - 2*np.outer(n, n)
        # T = np.eye(n.shape[0]) - 2 * (np.outer(x, x)/np.linalg.norm(x)**2)

        # Calculate new hamiltonian
        T.dot(H.dot(T, out=H), out=H)
        print('T: \n', T)
        H = np.delete(H, idx, axis=0)
        print('H after first delete: \n', H)
        H = np.delete(H, idx, axis=1)
        print('H after second delete: \n', H)
    return H


# Axel:
def UCC_ansatz(theta):
    vector = np.zeros(2**(theta.shape[0]))
    vector[[2**i for i in range(theta.shape[0])]] = theta
    return create_arbitrary_state(vector)


def matrix_to_pyquil(H):
    return qubitop_to_pyquilpauli(matrix_to_hamiltonian(H, 'JW'))


def matrix_to_hamiltonian(H, transform='none'):
    Hamiltonian = FermionOperator()

    if sparse.issparse(H):
        H_sparse = H.tocoo()
        for i, j, data in zip(H_sparse.row, H_sparse.col, H_sparse.data):
            Hamiltonian += data * FermionOperator(((int(i), 1), (int(j), 0)))

    else:
        for i in range(H.shape[0]):
            for j in range(H.shape[1]):
                Hamiltonian += H[i, j] * FermionOperator(((i, 1), (j, 0)))

    if transform == 'jordan' or transform == 'JW':
        return jordan_wigner(Hamiltonian)
    else:
        return Hamiltonian


def negative_energy_calculator(hamiltonian, initial_params, iterations=None):
    '''
    Calculates all negative eigenvalues of an hamiltonian matrix using vqe
    :param hamiltonian: np.array hamiltonian matrix
    :param initial_params: inital parameters for the ansatz
    :param iterations: desired amout of eigenvalues to find
    :return: a list of found energy eigenvalues
    '''
    if iterations is None:
        iterations = hamiltonian.shape[0]
    Energy = []
    for i in range(iterations):
        h_pyquil = matrix_to_pyquil(hamiltonian)
        result = vqe.vqe_run(UCC_ansatz, h_pyquil, initial_params, samples=None, qvm=qvm)
        Energy.append(result['fun'])
        if Energy[i] > 0:
            if iterations != hamiltonian.shape[0]:
                print('Warning: Unable to find the specified amount of eigenvalues')
            return Energy[:i]
        else:
            v = result['x'] # eigenvector
            hamiltonian = hamiltonian + 1.1 * np.abs(Energy[i]) * np.outer(v, v) # move found eigenvalue to > 0
    return Energy


def total_energy_calculator(hamiltonian, initial_params):
    '''
    Calculates all eigenvalues of an hamiltonian matrix using vqe
    :param hamiltonian: np.array hamiltonian matrix
    :param initial_params: inital parameters for the ansatz
    :return: a list of found energy eigenvalues
    '''
    Energy = negative_energy_calculator(hamiltonian, initial_params)
    if len(Energy) < hamiltonian.shape[0]:
        Energy = Energy + [-x for x in negative_energy_calculator(-1*hamiltonian, initial_params)]
    return Energy


# Calculate the energy:
# TODO: implement no_of_samples in vqe_run
no_of_samples = 1000


vqe = VQE(minimizer=minimize, minimizer_kwargs={'method': 'L-BFGS-B'})
qvm = QVMConnection()

j = 4
V = 1
OriginalHamiltonian = hamiltonian(j, V)[1].toarray()
TrueEigenvalues = eigenvalues(j, V)[1]
h = OriginalHamiltonian

print()
Energy = total_energy_calculator(h, [1, 0, 0, 0])
print('True eigenvalues: \n', TrueEigenvalues, '\n')
print('Calculated Energy: \n', Energy)








