import numpy as np
from pyquil.quil import Program
from grove.alpha.arbitrary_state.arbitrary_state import create_arbitrary_state
from openfermion import FermionOperator, QubitOperator, jordan_wigner
from forestopenfermion import qubitop_to_pyquilpauli
from pyquil.paulis import PauliSum, PauliTerm, exponential_map, trotterize, exponentiate
from scipy.optimize import minimize
from grove.pyvqe.vqe import VQE

term = FermionOperator(((0, 1), (1, 0))) - FermionOperator(((1, 1), (0, 0)))
term = jordan_wigner(term)
term = qubitop_to_pyquilpauli(term)
exp_map = exponential_map(-1j*term[0])
i = 0
print(exp_map(i))