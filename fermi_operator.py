#!/usr/bin/env python

"""Test script for the Fermi-Chebyshev expansion"""

import matplotlib.pyplot as plt
import numpy as np

from bodge import *

# Physical parameters.
U = {}

# List of physical parameters.
Lx = 30
Ly = 30
t = 1
μ = 0.1
Δ0 = 0.13 + 0.1j
m3 = 0.2

T = 1e-3 * t

# Construct the Hamiltonian.
lattice = CubicLattice((Lx, Ly, 1))
system = Hamiltonian(lattice)
fermi = FermiMatrix(system, 200)

U = np.zeros(lattice.shape)
for i in lattice.sites():
    # if (i[0] - L // 2) ** 2 + (i[1] - L // 2) ** 2 < (L // 3) ** 2:
    # U[i] = 1.6  # 1.1
    U[i] = 1.1

# Δ_old = 0.01
with system as (H, Δ):
    for i in lattice.sites():
        H[i, i] = -μ * σ0

    for i, j in lattice.bonds():
        H[i, j] = -t * σ0

# Δ_init = 0.04
# with system as (H, Δ):
#     for i in lattice.sites():
#         Δ[i, i] = Δ_init * jσ2

# Determine initial guess via geometric binary search.
# Δ_init = 0.0606

# with system as (H, Δ):
#     for i in lattice.sites():
#         Δ[i, i] = Δ_init * jσ2
Δ_min = 1e-6
Δ_max = 1
for n in range(6):
    Δ_init = np.sqrt(Δ_min * Δ_max)

    with system as (H, Δ):
        for i in lattice.sites():
            if U[i] != 0:
                Δ[i, i] = Δ_init * jσ2

    Δ_diff = fermi(T).order_swave(U)

    new = 0
    old = 0
    for i in lattice.sites():
        if U[i] != 0:
            new += Δ_diff[i]
            old += Δ_init

    if np.abs(new) > old:
        Δ_min = Δ_init
    else:
        Δ_max = Δ_init

    print(Δ_min, Δ_init, Δ_max)

# Construct the Fermi matrix.
Δs = [Δ_init]
for n in range(1, 100):
    Δ_new = fermi(T).order_swave(U)
    Δs.append(np.mean(Δ_new))

    # Steffensens method
    # if n % 6 == 0:
    #     Δs.append(Δs[-3] - (Δs[-2] - Δs[-3]) ** 2 / (Δs[-1] - 2 * Δs[-2] + Δs[-3]))
    #     diff = np.mean(Δs[-1] / Δs[-3])
    # else:
    #     diff = Δs[-1] / Δs[-2]

    diff = Δs[-1] / Δs[-2]

    with system as (H, Δ):
        for i in lattice.sites():
            Δ[i, i] = Δ_new[i] * jσ2

    plt.figure()
    plt.plot(np.abs(Δ_new)[:, Ly // 2, 0])
    plt.ylim([0, None])
    # plt.imshow(np.abs(Δ_new), vmin=0)
    # plt.colorbar()
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
