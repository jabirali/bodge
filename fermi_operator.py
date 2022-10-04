#!/usr/bin/env python

"""Test script for the Fermi-Chebyshev expansion"""

import matplotlib.pyplot as plt
import numpy as np

from bodge import *

# Physical parameters.
U = {}

# List of physical parameters.
L = 20
t = 1
μ = 0.1
Δ0 = 0.13 + 0.1j
m3 = 0.2

T = 1e-4 * t

# Construct the Hamiltonian.
lattice = CubicLattice((L, L, 1))
system = Hamiltonian(lattice)
fermi = FermiMatrix(system, 200)

U = np.zeros(lattice.shape)
for i in lattice.sites():
    # if i[0] <= L // 2:
    U[i] = 1

Δ_old = 0.01
with system as (H, Δ):
    for i in lattice.sites():
        H[i, i] = -μ * σ0

    for i, j in lattice.bonds():
        H[i, j] = -t * σ0

# Determine initial guess.
Δ_min = 1e-16
Δ_max = 1
for n in range(10):
    Δ_init = (Δ_min + Δ_max) / 2

    with system as (H, Δ):
        for i in lattice.sites():
            Δ[i, i] = Δ_init * jσ2

    Δ_diff = fermi(T).order_swave(U)
    if np.abs(np.sum(Δ_diff * U)) > np.abs(np.sum(U)):
        Δ_min = Δ_init
    else:
        Δ_max = Δ_init

    print(Δ_min, Δ_init, Δ_max)

# Construct the Fermi matrix.
Δs = [Δ_init]
for n in range(1, 100):
    # TODO: Test binary search for correct initial Δ(U).
    Δ_new = fermi(T).order_swave(U)
    Δs.append(np.mean(Δ_new))

    # if n % 8 == 0:
    #     Δs.append(Δs[-3] - (Δs[-2] - Δs[-3]) ** 2 / (Δs[-1] - 2 * Δs[-2] + Δs[-3]))
    #     diff = np.mean(Δs[-1] / Δs[-3])
    # else:
    diff = Δs[-1] / Δs[-2]

    with system as (H, Δ):
        for i in lattice.sites():
            Δ[i, i] = Δ_new[i] * jσ2

    plt.figure()
    plt.imshow(np.abs(Δ_new))
    plt.colorbar()
    plt.show()
    print(n, np.sign(np.real(diff)), np.abs(diff), np.mean(Δs[-1]))


# Δs = fermi(T).order_swave(U)
# print(Δs / 1e-10)


# plt.figure()
# plt.imshow(np.abs(U))
# plt.colorbar()

# plt.figure()
# plt.imshow(fermi.current_elec(axis=1))
# plt.colorbar()

# plt.show()
