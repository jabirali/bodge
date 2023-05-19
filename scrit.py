#!/usr/bin/env python

"""Calculation of critical temperature for s-wave superconductors."""

from time import time
import numpy as np
from icecream import ic

from bodge import *

# ------------------------------------------------------------
# Specify the physical system under investigation.
# ------------------------------------------------------------

with open("bench.dat", "w") as f:
    for N in [1, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600, 2800, 3000, 3200, 3400, 3600, 3800, 4000]:
        # Model parameters.
        Lx = 32
        Ly = 32
        Lz = 1 # 2

        t = 1.0
        μ = 0.0
        U = t

        # Construct an appropriate lattice, including functions to determine
        # whether a particular region is superconducting or normal metallic.
        lattice = CubicLattice((Lx, Ly, Lz))

        def NM(i: Coord):
            x, y, z = i
            return z == 1

        def SC(i: Coord):
            x, y, z = i
            return z == 0 # and x >= Lx//4 and x < 3*Lx//4 and y >= Ly//4 and y < 3*Ly//4

        def IN(i: Coord):
            return NM(i) or SC(i)

        # Calculate critical temperature.
        for τ in [0]: # np.arange(0, 0.1, 0.01):
            system = Hamiltonian(lattice)
            with system as (H, Δ, V):
                for i in lattice.sites():
                    # Chemical potential in non-empty space.
                    if IN(i):
                        H[i, i] = -μ * σ0

                    # Attractive Hubbard in superconductors.
                    if SC(i):
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
            Tc = critical_temperature(system, order=N, T_max=0.2)
            sec = time() - sec

            f.write(f"{N}, {sec}, {Tc}\n")
            f.flush()

