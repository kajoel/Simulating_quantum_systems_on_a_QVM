import numpy as np
import copy

N = 3
psi_ex = np.array([1., np.NaN, 0., 0.])


class State:

    def __init__(self, N=2, psi=np.full(2, np.NaN)):
        self.N = N
        self.psi = psi


def dot(psi1, psi2):
    if np.array_equal(psi1, psi2):
        return 1
    else:
        return 0


psi_test = State(2, psi_ex)


def a_annihilate(psi, p, sigma):
    psi_tmp = copy.deepcopy(psi)
    if (sigma == 1 and psi.psi[p] == 1) or (sigma == -1 and psi.psi[p] == 0):
        psi_tmp.psi[p] = np.NaN
    else:
        # raise ValueError('Cant create state')
        psi_tmp.psi[p] = 2
    return psi_tmp


def a_create(psi, p, sigma):
    psi_tmp = copy.deepcopy(psi)
    if sigma == 1 and np.isnan(psi.psi[p]):
        psi_tmp.psi[p] = 1
    elif sigma == -1 and np.isnan(psi.psi[p]):
        psi_tmp.psi[p] = 0
    else:
        psi_tmp.psi[p] = 2
    return psi_tmp


def j_plus(psi, p):
    try:
        return a_create(a_annihilate(psi, p, -1), p, 1)
    except ValueError:
        return ValueError


def j_minus(psi, p):
    try:
        return a_create(a_annihilate(psi, p, 1), p, -1)
    except ValueError:
        return ValueError


print(psi_test.psi)
print(a_create(psi_test, 1, 1).psi)
print(j_plus(j_plus(psi_test, 3), 2).psi + j_plus(j_plus(psi_test, 3), 2).psi)
print(dot(np.array([np.Inf, np.Inf]), np.array([np.Inf, np.Inf])))


def hamiltonian_kinetic(psi, p, sigma):
    try:
        return a_create(a_annihilate(psi, p, sigma), p, sigma)
    except ValueError:
        return ValueError


def hamiltonian_potential(psi, p, q):
    try:
        return State(j_plus(j_plus(psi, p), q).psi + j_minus(j_minus(psi, p), q).psi)
    except ValueError:
        return ValueError


H = np.zeros((2**N, 2**N))
states = []

for i in range(0, (2**N)):
    states.append(State(N, np.fromstring(" ".join(bin(i)[2:].rjust(N, '0')), dtype=float, sep=' ')))

    # converts to binary, removes 0b, separates, converts to np.array

# print(j_plus(j_plus(states[:, 1], 0), 0))

# Calculate Kinetic Hamiltonian:
for i in range(0, 2**N):
        index = 0
        for p in range(0, N):
            try:
                index += -1*dot(states[i].psi, hamiltonian_kinetic(states[i], p, -1).psi)
            except ValueError:
                index += 0
            try:
                index += dot(states[i].psi, hamiltonian_kinetic(states[i], p, 1).psi)
            except ValueError:
                index += 0
        H[i, i] = index

# Calculate exchange:
for i in range(0, 2**N):
    for j in range(0, 2**N):
        index = 0
        for p in range(0, N):
            for q in range(0, N):
                try:
                    index += (dot(states[i].psi, j_plus(j_plus(states[j], p), q).psi) + dot(states[i].psi, j_minus(j_minus(states[j], p), q).psi))
                    # print(index)
                except ValueError:
                    index += 0
        H[i, j] += index


print('\n Hamiltonian: \n', H)
ev, evec = np.linalg.eig(H)
print('\n Eigenvalues: \n', ev)
