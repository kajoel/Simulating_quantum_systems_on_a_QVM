from openfermion.transforms import jordan_wigner,get_sparse_operator, get_fermion_operator
from openfermion.ops import FermionOperator, QubitOperator
from forestopenfermion import pyquilpauli_to_qubitop, qubitop_to_pyquilpauli
import numpy as np
from scipy.linalg import eigh
from pyquil.paulis import sZ, exponentiate
from pyquil.quil import Program
import pyquil.api as api
from pyquil.gates import *
from grove.pyvqe.vqe import VQE
from grove.alpha.arbitrary_state.arbitrary_state import create_arbitrary_state
from grove.alpha.arbitrary_state.unitary_operator import fix_norm_and_length
from scipy.optimize import minimize
import matplotlib.pyplot as plt

#GROVE
vqe = VQE(minimizer=minimize,minimizer_kwargs={'method': 'nelder-mead'})
pauli_channel = [0.1, 0.1, 0.1]
qvm = api.QVMConnection()
#---------------------------------------------------------------------------------
h = np.array([[-1.5, np.sqrt(3)],[np.sqrt(3), 0.5]])
H = FermionOperator()

for i,j in np.ndindex(h.shape):
    H += h[i,j] * FermionOperator(((i, 1), (j, 0)))

#Jordan-Wigner transform of H
H_JW = jordan_wigner(H)

#From OpenFermions to PyQuil
H_pyquil = qubitop_to_pyquilpauli(H_JW)

H_eig = get_sparse_operator(H_JW).todense()
eigvalues, eigvectors = eigh(h)
print('Eigeinvalue\n')
print(eigvalues)
print('Eigenvectors\n')
print(eigvectors)

#EXAMPLES:
#--------------------------------------------------------------------------------
#SMALL ANSATZ
def small_ansatz(params):
    return Program(RX(params[0], 0))

# Our Hamiltonian is just \sigma_z on the zeroth qubit
hamiltonian = sZ(0)
initial_angle = [0.0]

result = vqe.vqe_run(small_ansatz, hamiltonian, initial_angle, 10000, qvm=qvm)
#print('\n\n')
#print('Result (small_ansatz):')
#print(result)
#print('\n\n')

#------------------------------------------------------------------------------------------------
#More Sophisticated Ansatzes

def smallish_ansatz(params):
    return Program(RX(params[0], 0), RX(params[1], 0))

initial_angles = [1.0, 1.0]
result = vqe.vqe_run(smallish_ansatz, hamiltonian, initial_angles, 10000, qvm=qvm)
#print('\n\n')
#print('Result (smallish_ansatz):')
#print(result)
#print('\n\n')

#New ansatz

def variable_gate_ansatz(params):
    gate_num = int(np.round(params[1])) # for scipy.minimize params must be floats
    p = Program(RX(params[0], 0))
    for gate in range(gate_num):
        p.inst(X(0))
    return p

print(variable_gate_ansatz([0.5, 3]))

initial_params = [1.0, 3]
#result = vqe.vqe_run(variable_gate_ansatz, hamiltonian, initial_params, 10000, qvm=qvm)
#print(result)

def UCC_ansatz(theta):
    p = Program(
        X(0),
        X(1),
        RX(-np.pi/2, 0),
        RY(np.pi/2, 1),
        CNOT(0, 1),
        RZ(theta[0], 1),
        CNOT(0, 1),
        RX(np.pi/2, 0),
        RY(-np.pi/2, 1))
    return p

#create_arbitrary_state(vector, qubits=None)
#Number of qubites
n = 4
vector = np.zeros([2**n])
vector[[1]] = [1]

def UCC_ansatz(theta):
    #theta = fix_norm_and_length(theta)
    vector = np.zeros(2**(theta.shape[0]))
    vector[[2**i for i in range(theta.shape[0])]] = theta
    return create_arbitrary_state(vector)

result = vqe.vqe_run(UCC_ansatz, H_pyquil, [1,0], 100000, qvm=qvm)
print(result)

wfn = api.WavefunctionSimulator().wavefunction(UCC_ansatz(result['x']))
print(wfn)