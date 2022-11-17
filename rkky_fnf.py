#!/usr/bin/env python

"""RKKY interactions between two ferromagnets separated by a normal spacer."""

import numpy as np

from bodge import *
from bodge.common import *
from bodge.utils import ldos

# Tight-binding parameters.
t = 1
μ = -3
M = t/2

# Construct a 2D lattice.
Lx = 100
Ly = 100
Lz = 1

lattice = CubicLattice((Lx, Ly, Lz))

# Compare ferromagnetic and antiferromagnetic alignments
# of the two magnetic layers for various separations δ.
# Similarly to Deaven et al. (1991), we consider an
# oblate Fermi surface with t_x < t_y here.
for δ in range(1, 21):
    # Construct the ferromagnetic Hamiltonian.
    system = Hamiltonian(lattice)
    with system as (H, Δ, V):
        for i in lattice.sites():
            if i[0] >= 0 and i[0] <= 2:
                H[i, i] = -μ * σ0 + M * σ3
            elif i[0] >= 3 + δ and i[0] <= 5 + δ:
                H[i, i] = -μ * σ0 + M * σ3
            else:
                H[i, i] = -μ * σ0

        for i, j in lattice.bonds(axis=0):
            H[i, j] = -0.8 * t * σ0

        for i, j in lattice.bonds(axis=1):
            H[i, j] = -t * σ0

    # Calculate the corresponding free energy.
    E_fm = free_energy(system)

    # Construct the antiferromagnetic Hamiltonian.
    with system as (H, Δ, V):
        for i in lattice.sites():
            if i[0] == 3 + δ or i[0] == 4 + δ or i[0] == 5 + δ:
                H[i, i] = -μ * σ0 - M * σ3

    # Calculate the corresponding free energy.
    E_afm = free_energy(system)

    # Print the results in CSV format.
    print(f"{δ}, {(E_fm - E_afm)/(Ly*Lz)}")
