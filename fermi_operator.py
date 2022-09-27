#!/usr/bin/env python

"""Test script for the Fermi-Chebyshev expansion"""

import matplotlib.pyplot as plt
import numpy as np

from bodge import *

# List of physical parameters.
t = 1.0
μ = 0.1 * t
Δ0 = (0.13 + 0.1j) * t
m3 = 0.2 * t

# Construct the Hamiltonian.
lattice = CubicLattice((40, 40, 1))
system = Hamiltonian(lattice)

with system as (H, Δ):
    for i in lattice.sites():
        if i[0] < 20:
            H[i, i] = -μ * σ0
            Δ[i, i] = Δ0 * jσ2
        else:
            H[i, i] = -μ * σ0 - m3 * σ3

    for i, j in lattice.bonds():
        H[i, j] = -t * σ0

# Construct the Fermi matrix.
fermi = FermiMatrix(system, 30)
fermi(0.05, 32)

print(fermi.gap_ss())

# for i in lattice.sites():
#     if i[1] == 20:
#         print((-system.scale / 2) * np.trace(f[i, i] @ jσ2) / Δ0)
# print(g)

# Δ = system.scale * F.diagonal(3)[::4]

# plt.plot(np.real(Δ))
# plt.show()
