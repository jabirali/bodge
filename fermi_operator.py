#!/usr/bin/env python

"""Test script for the Fermi operator expansion"""

from bodge import *

# List of physical parameters.
t = 1.0
μ = 0.1 * t
Δ0 = 0.1 * t
m3 = 0.2 * t

# Construct the Hamiltonian.
lattice = CubicLattice((20, 20, 2))
system = Hamiltonian(lattice)

with system as (H, Δ):
    for i in lattice.sites():
        if i[0] < 10:
            H[i, i] = -μ * σ0
            Δ[i, i] = Δ0 * jσ2
        else:
            H[i, i] = -μ * σ0 - m3 * σ3

    for i, j in lattice.bonds():
        H[i, j] = -t * σ0

# Test the WIP code.
from bodge.math import *

for T_n in chebyshev(system):
    print(T_n)
