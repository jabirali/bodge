#!/usr/bin/env python

"""Calculation of critical temperature for s-wave superconductors."""

from time import time

import numpy as np
from icecream import ic

from bodge import *

# ------------------------------------------------------------
# Specify the physical system under investigation.
# ------------------------------------------------------------

with open("scrit.dat", "w") as f:
    # Model parameters.
    Lx = 64
    Ly = 64
    Lz = 2

    t = 1.0
    μ = 0.5
    m = 2.0
    U = t

    # Construct an appropriate lattice, including functions to determine
    # whether a particular region is superconducting or normal metallic.
    lattice = CubicLattice((Lx, Ly, Lz))

    def NM(i: Coord):
        x, y, z = i
        return z == 1

    def SC(i: Coord):
        x, y, z = i
        return z == 0 and x >= Lx // 4 and x < 3 * Lx // 4 and y >= Ly // 4 and y < 3 * Ly // 4

    def IN(i: Coord):
        return NM(i) or SC(i)

    # Calculate critical temperature.
    for τ in [0.05]:  # np.arange(0, 0.1, 0.01):
        system = Hamiltonian(lattice)
        with system as (H, Δ, V):
            for i in lattice.sites():
                # Chemical potential in non-empty space,
                # exchange field in non-superconductors.
                # Attractive Hubbard in superconductors.
                if NM(i):
                    H[i, i] = -μ * σ0 - m * σ3
                if SC(i):
                    H[i, i] = -μ * σ0
                    V[i, i] = -U

            # Intra-plane hopping coefficient t.
            for i, j in lattice.bonds(axis=0):
                if IN(i) and IN(j):
                    H[i, j] = -t * σ0
            for i, j in lattice.bonds(axis=1):
                if IN(i) and IN(j):
                    H[i, j] = -t * σ0

            # Inter-plane hopping coefficient τ.
            for i, j in lattice.bonds(axis=2):
                if IN(i) and IN(j):
                    H[i, j] = -τ * σ0

        # TODO: Number of iterations per temp?
        sec = time()
        Tc = critical_temperature(system, T_max=0.04)
        sec = time() - sec

        f.write(f"{sec}, {Tc}\n")
        f.flush()
