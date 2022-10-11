#!/usr/bin/env python

"""Calculation of critical temperature for s-wave superconductors."""

import numpy as np
from tqdm import tqdm, trange

from bodge import *

# ------------------------------------------------------------
# Specify the physical system under investigation.
# ------------------------------------------------------------

# Physical parameters.
Lx = 60
Ly = 20

t = 1
μ = 0.1
U = 1.5

# Numerical parameters.
N = 200
R = None

# Non-superconducting Hamiltonian.
lattice = CubicLattice((Lx, Ly, 1))
system = Hamiltonian(lattice)
fermi = FermiMatrix(system, N)

with system as (H, Δ, V):
    for i in lattice.sites():
        H[i, i] = -μ * σ0
        V[i, i] = -U

    for i, j in lattice.bonds():
        H[i, j] = -t * σ0

# ------------------------------------------------------------
# Determine Δ0 via geometric binary search.
# ------------------------------------------------------------

T = 1e-10
Δ_min = 1e-6
Δ_max = 1

for n in trange(12, desc="Δ0", unit="val"):
    # Hamiltonian update.
    Δ0 = np.sqrt(Δ_min * Δ_max)
    with system as (H, Δ, V):
        for i in lattice.sites():
            if V[i, i] != 0:
                Δ[i, i] = Δ0 * jσ2

    # Convergence control.
    F = fermi(T)
    Δ2 = np.abs(F.order_swave())
    Δ1 = np.where(Δ2 > 0, Δ0, 0)

    if np.mean(Δ2) > np.mean(Δ1):
        Δ_min = Δ0
    else:
        Δ_max = Δ0

    print(f"Δ0: {Δ0}")

# ------------------------------------------------------------
# Determine Tc via regular binary search.
# ------------------------------------------------------------

δ = Δ0 * 1e-4
T_min = 1e-10
T_max = 2 * (Δ0 / 1.764)
Tc = (T_min + T_max) / 2

with system as (H, Δ, V):
    for i in lattice.sites():
        if V[i, i] != 0:
            Δ[i, i] = δ * jσ2

for n in trange(12, desc="Tc", unit="val"):
    # Convergence control.
    F = fermi(T)
    Δ2 = np.abs(F.order_swave())
    Δ1 = np.where(Δ2 > 0, δ, 0)

    # Temperature update.
    if np.mean(Δ2) > np.mean(Δ1):
        T_min = Tc
    else:
        T_max = Tc
    Tc = (T_min + T_max) / 2

    print(f"Tc: {Tc}")
