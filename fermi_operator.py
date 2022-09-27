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
lattice = CubicLattice((30, 30, 1))
system = Hamiltonian(lattice)

with system as (H, Δ):
    for i in lattice.sites():
        if i[0] < 15:
            H[i, i] = -μ * σ0
            Δ[i, i] = Δ0 * np.exp(1j * 0.01 * i[1]) * jσ2
        else:
            H[i, i] = -μ * σ0 - m3 * σ3

    for i, j in lattice.bonds():
        H[i, j] = -t * σ0

# Construct the Fermi matrix.
fermi = FermiMatrix(system, 30)
fermi(0.05, 32)

plt.figure()
plt.imshow(np.abs(fermi.order_swave()))
plt.colorbar()

plt.figure()
plt.imshow(fermi.current_elec(1))
plt.colorbar()

plt.show()
