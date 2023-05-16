#!/usr/bin/env python

"""Calculation of critical temperature for s-wave superconductors."""

import numpy as np

from bodge import *

# ------------------------------------------------------------
# Specify the physical system under investigation.
# ------------------------------------------------------------

# Physical parameters.
N = 8000
Lx = 20
Ly = 20

t = 1.0
μ = 0.0
U = t / 3

# Non-superconducting Hamiltonian.
lattice = CubicLattice((Lx, Ly, 1))
system = Hamiltonian(lattice)

with system as (H, Δ, V):
    for i in lattice.sites():
        H[i, i] = -μ * σ0
        V[i, i] = -U

    for i, j in lattice.bonds():
        H[i, j] = -t * σ0

Tc = critical_temperature(system, N)
print(Tc)
