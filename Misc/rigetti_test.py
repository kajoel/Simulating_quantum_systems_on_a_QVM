#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test-script for Rigetti software (QVM and QuilC)
First, make sure that QVM and QuilC are running in server mode, start them
with qvm -S and quilc -R or quilc -S.

Created on Sat Feb 16 15:21:03 2019
@author: kajoel
"""

#%% Input args to get_qc (for launching qc)

# NOTE: use get_qc([n]q-qvm) (where [n] is replaced with the number of
# qubits) to get a fully conected qc (the topology will be a complete
# graph)

from pyquil import list_quantum_computers
list_quantum_computers()

#%% Quantum Bell state (entaglement) example

# This example doesn't work with quilc -R, however, it works fine with the
# outdated quilc -S

# Imports:
import pyquil.gates as qg       # qc is short for quantum_gate
from pyquil import Program, get_qc

# Construct a Bell State program
p = Program(qg.H(0), qg.CNOT(0, 1))

# Run the program on a QVM
qc = get_qc('9q-square-qvm')
result = qc.run_and_measure(p, trials=20)

# The output is a dictionary with integer keys 0-8 representing the qubits.
# Default start-value 0 and since no gates affect qb 2-8 these stay 0.
print(result[0])
print(result[1])

#%% Parametric example

# An example of a parametric program which is first created and compiled
# (once) and then used multiple times. This is useful in hybrid-
# algorithms

# Imports:
import numpy as np
import pyquil.gates as qg       # qc is short for quantum_gate
from pyquil import Program, get_qc
from scipy.optimize import minimize
from matplotlib import pyplot as plt


def ansatz():
    program = Program()
    theta = program.declare('theta', memory_type='REAL')
    ro = program.declare('ro', memory_type='BIT', memory_size=1)
    program += qg.RY(theta, 0)
    program += qg.MEASURE(0, ro[0])
    return program


# Create executable
qc = get_qc("9q-square-qvm")
program = ansatz()  # look ma, no arguments!
program.wrap_in_numshots_loop(shots=1000)
executable = qc.compile(program)

# ##### Parameter sweep #####
thetas = np.linspace(0, 2*np.pi, 21)
results = []
for theta in thetas:
    bitstrings = qc.run(executable, memory_map={'theta': [theta]})
    results.append(np.mean(bitstrings[:, 0]))

# Plot
plt.plot(thetas, results, 'o-')
plt.xlabel(r'$\theta$', fontsize=18)
_ = plt.ylabel(r'$\langle \Psi(\theta) | \frac{1 - Z}{2} | \Psi(\theta) \rangle$', fontsize=18)

# ##### Optimization #####
def objective_function(thetas):
    bitstrings = qc.run(executable, memory_map={'theta': thetas})
    result = np.mean(bitstrings[:, 0])
    return -result


res = minimize(objective_function, x0=[0.1], method='COBYLA')

# Plot
plt.plot(thetas, results, label='scan')
plt.plot([res.x], [-res.fun], '*', ms=20, label='optimization result')
plt.legend()
plt.show()
