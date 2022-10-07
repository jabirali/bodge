#!/usr/bin/env python

"""Self-consistent calculation for s-wave superconductors."""

import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange

from bodge import *

# List of physical parameters.
Lx = 32
Ly = 32

t = 1
μ = 0.1
U = 1.5

R = None
T = 1e-6 * t

# Construct the Hamiltonian.
lattice = CubicLattice((Lx, Ly, 1))
system = Hamiltonian(lattice)
fermi = FermiMatrix(system, 200)

with system as (H, Δ, V):
    for i in lattice.sites():
        H[i, i] = -μ * σ0
        V[i, i] = -U

    for i, j in lattice.bonds():
        H[i, j] = -t * σ0

# Determine initial guess via geometric binary search.
Δ_min = 1e-6
Δ_max = 1
for n in trange(6, desc="boot", unit="cyc", leave=False):
    # Hamiltonian update.
    Δ_init = np.sqrt(Δ_min * Δ_max)
    with system as (H, Δ, V):
        for i in lattice.sites():
            Δ[i, i] = Δ_init * jσ2

    # Convergence control.
    Δ2 = np.abs(fermi(T).order_swave())
    Δ1 = np.where(Δ2 > 0, Δ_init, 0)

    if np.mean(Δ2) > np.mean(Δ1):
        Δ_min = Δ_init
    else:
        Δ_max = Δ_init

# Convergence via accelerated self-consistency iteration.
Δs = [Δ_init]
for n in trange(100, desc="conv", unit="cyc", leave=False):
    # Convergence control and acceleration.
    if len(Δs) > 3:
        if diff < 1e-6:
            break
        else:
            Δs = [Δs[-3] - (Δs[-2] - Δs[-3]) ** 2 / (Δs[-1] - 2 * Δs[-2] + Δs[-3])]

    # Hamiltonian update.
    with system as (H, Δ, V):
        for i in lattice.sites():
            Δ[i, i] = Δs[-1][i] * jσ2

    # Order parameter update.
    Δs.append(fermi(T, R).order_swave())
    diff = np.mean(np.abs(1 - Δs[-1] / Δs[-2]))

    # Status information.
    print()
    print(f"Gap: {np.real(np.mean(Δs[-1]))}")
    print(f"Diff: {np.real(np.mean(diff))}")

    # Status plot.
    plt.figure()
    plt.imshow(np.abs(Δs[-1]), vmin=0)
    plt.colorbar()
    plt.show()
