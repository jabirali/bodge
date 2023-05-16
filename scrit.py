#!/usr/bin/env python

"""Calculation of critical temperature for s-wave superconductors."""

import numpy as np

from bodge import *

# ------------------------------------------------------------
# Specify the physical system under investigation.
# ------------------------------------------------------------

# Physical parameters.
N = 1000
# TODO: 64x64 normal metal under
Lx = 32
Ly = 32

t = 1.0
μ = 0.0
U = t

for τ in [0.001 * t, 0.003 * t, 0.01 * t, 0.03 * t, 0.10 * t, 0.30 * t, 1.00 * t]:

    # Non-superconducting Hamiltonian.
    #lattice = CubicLattice((Lx, Ly, 1))
    lattice = CubicLattice((Lx, Ly, 2))
    system = Hamiltonian(lattice)

    with system as (H, Δ, V):
        for i in lattice.sites():
            H[i, i] = -μ * σ0
            if i[2] == 0:
                V[i, i] = -U
            else:
                V[i, i] = 0

        for i, j in lattice.bonds(axis=0):
            H[i, j] = -t * σ0
        for i, j in lattice.bonds(axis=1):
            H[i, j] = -t * σ0
        for i, j in lattice.bonds(axis=2):
            H[i, j] = -τ * σ0

    Tc = critical_temperature(system, order=N, T_max=0.2)
    print(":: ", τ, Tc)
