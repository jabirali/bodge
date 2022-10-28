#!/usr/bin/env python


import csv
from time import time

# import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
from scipy.linalg import eigh, solve
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import eigsh, spsolve
from tqdm import tqdm, trange

from bodge import *
from bodge.utils import ldos

Lx = 100
Ly = 100

t = 1
μ = -0.1

Δ_0 = 0.1 * t

lattice = CubicLattice((Lx, Ly, 1))
system = Hamiltonian(lattice)

# d = dvector("(e_x + je_y) * p_x")
# d = dvector("e_z * (p_x + jp_y)")
d = dvector("e_z * (p_x + jp_y)")

with system as (H, Δ, V):
    for i in lattice.sites():
        H[i, i] = -μ * σ0
        # Δ[i, i] = -Δ_0 * jσ2

    for i, j in lattice.bonds():
        H[i, j] = -t * σ0
        # Δ[i, j] = -Δ_0 * d(i, j) / 2

sites = [
    # (0, Ly // 2, 0),
    # (Lx // 2, 0, 0),
    (Lx // 2, Ly // 2, 0),
]

energies = np.linspace(0, 2 * Δ_0, 20)
# energies = [0, 0.001]

t = time()

dos = ldos(system, sites, energies, 0.3 * Δ_0)

# print(dos.keys())
# plt.figure(
# plt.plot(energies, dos.values())
# plt.show()

print("\n", time() - t, "s")
