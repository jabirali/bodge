#!/usr/bin/env python

"""Test script for the Fermi-Chebyshev expansion"""

import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange

from bodge import *

# Physical parameters.
U = {}

# List of physical parameters.
Lx = 50
Ly = 20

t = 1
μ = 0.1

R = None
T = 1e-10 * t

# Construct the Hamiltonian.
lattice = CubicLattice((Lx, Ly, 1))
system = Hamiltonian(lattice)
fermi = FermiMatrix(system, 200)

U = np.zeros(lattice.shape)
for i in lattice.sites():
    # if (i[0] - L // 2) ** 2 + (i[1] - L // 2) ** 2 < (L // 3) ** 2:
    # U[i] = 1.6  # 1.1
    U[i] = 1.5

# Δ_old = 0.01
with system as (H, Δ):
    for i in lattice.sites():
        H[i, i] = -μ * σ0

    for i, j in lattice.bonds():
        H[i, j] = -t * σ0

# Δ_init = 0.2065
# with system as (H, Δ):
#     for i in lattice.sites():
#         Δ[i, i] = Δ_init * jσ2

# Determine initial guess via geometric binary search.
Δ_min = 1e-6
Δ_max = 1
for n in trange(6, desc="boot", unit="cyc", leave=False):
    # Hamiltonian update.
    Δ_init = np.sqrt(Δ_min * Δ_max)
    with system as (H, Δ):
        for i in lattice.sites():
            if U[i] != 0:
                Δ[i, i] = Δ_init * jσ2

    # Order parameter update.
    Δ_diff = fermi(T).order_swave(U)

    # Convergence control.
    if np.sum(np.abs(U * Δ_diff)) > np.sum(np.abs(U * Δ_init)):
        Δ_min = Δ_init
    else:
        Δ_max = Δ_init

# Construct the Fermi matrix.
Δs = [Δ_init * np.ones_like(U)]
for n in trange(100, desc="conv", unit="cyc", leave=False):
    # Hamiltonian update.
    with system as (H, Δ):
        for i in lattice.sites():
            Δ[i, i] = Δs[-1][i] * jσ2

    # Order parameter update.
    Δs.append(fermi(T, R).order_swave(U))
    diff = np.mean(np.abs(1 - Δs[-1] / Δs[-2]))

    # Convergence control and acceleration.
    if len(Δs) > 3:
        if diff < 1e-6:
            break
        else:
            Δs[-1] = Δs[-3] - (Δs[-2] - Δs[-3]) ** 2 / (Δs[-1] - 2 * Δs[-2] + Δs[-3])
        Δs = Δs[-1:]

    # Status information.
    print()
    print(f"Gap: {np.real(np.mean(Δs[-1]))}")
    print(f"Diff: {np.real(np.mean(diff))}")

    # Status plot.
    plt.figure()
    plt.imshow(np.abs(Δs[-1]), vmin=0)
    plt.colorbar()
    plt.show()


# Δs = fermi(T).order_swave(U)
# print(Δs / 1e-10)


# plt.figure()
# plt.imshow(np.abs(U))
# plt.colorbar()

# plt.figure()
# plt.imshow(fermi.current_elec(axis=1))
# plt.colorbar()

# plt.show()
