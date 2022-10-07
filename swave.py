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

Δ_init = 0.2065
with system as (H, Δ):
    for i in lattice.sites():
        Δ[i, i] = Δ_init * jσ2

# Determine initial guess via geometric binary search.

# with system as (H, Δ):
#     for i in lattice.sites():
#         Δ[i, i] = Δ_init * jσ2
# Δ_min = 1e-6
# Δ_max = 1
# for n in trange(6, desc="Δ bootstrap", leave=False):
#     Δ_init = np.sqrt(Δ_min * Δ_max)

#     with system as (H, Δ):
#         for i in lattice.sites():
#             if U[i] != 0:
#                 Δ[i, i] = Δ_init * jσ2

#     Δ_diff = fermi(T).order_swave(U)

#     new = 0
#     old = 0
#     for i in lattice.sites():
#         if U[i] != 0:
#             new += Δ_diff[i]
#             old += Δ_init

#     if np.abs(new) > old:
#         Δ_min = Δ_init
#     else:
#         Δ_max = Δ_init

# print(Δ_min, Δ_init, Δ_max)

# Construct the Fermi matrix.
Δs = [Δ_init]
Δa = Δ_init
α = 0.01
for n in trange(1, 100, desc="Δ converge", leave=False):
    # Convergence acceleration method.
    Δs.append(fermi(T, 20).order_swave(U))
    if n % 4 == 0:
        Δa = Δs[-3] - (Δs[-2] - Δs[-3]) ** 2 / (Δs[-1] - 2 * Δs[-2] + Δs[-3])
    else:
        Δa = Δs[-1] * α + Δa * (1 - α)

    diff = 1 - Δs[-1] / Δs[-2]

    with system as (H, Δ):
        for i in lattice.sites():
            Δ[i, i] = Δa[i] * jσ2

    print()
    print(f"Gap: {np.real(np.mean(Δs[-1]))}")
    print(f"Diff: {np.real(np.mean(diff))}")

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
