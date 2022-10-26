#!/usr/bin/env python

import csv
from time import time

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
from scipy.linalg import eigh, solve
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import eigsh, spsolve
from tqdm import tqdm, trange

from bodge import *

Lx = 100
Ly = 100

t = 1
μ = 0

Δ_0 = 0.1 * t

lattice = CubicLattice((Lx, Ly, 1))
system = Hamiltonian(lattice)

d = dvector("e_x * (p_x + jp_y)")

with system as (H, Δ, V):
    for i in lattice.sites():
        H[i, i] = -μ * σ0
        # Δ[i, i] = -Δ_0 * jσ2

    for i, j in lattice.bonds():
        H[i, j] = -t * σ0
        Δ[i, j] = -Δ_0 * d(i, j)

# D = np.zeros(20)
# for ε in np.linalg()

i0 = lattice[(0, Ly // 2, 0)]
i1 = lattice[(Lx // 2, 0, 0)]
i2 = lattice[(Lx // 2, Ly // 2, 0)]

H = system.scale * system.compile()
I = system.identity.tocsr()

ε = 0.1 * Δ_0
η = 1e-3 * Δ_0
Hz = (-1j / π) * ((ε + 1j * η) * I - H)


t = time()

for ii in [i0, i1, i2]:
    e = coo_matrix(([1], ([4 * ii], [0])), shape=(H.shape[1], 1)).tocsr()
    print(np.real(spsolve(Hz, e)[4 * ii]))


print("\n", time() - t, "s")
