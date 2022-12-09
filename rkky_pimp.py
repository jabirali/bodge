#!/usr/bin/env python

"""RKKY interactions between two magnetic impurities."""

import numpy as np

from bodge import *
from bodge.common import *

# Tight-binding parameters.
t = 1
μ = -t
M = +3 * t / 2

# Construct a 2D lattice.
Lx = 64
Ly = 64
Lz = 1

lattice = CubicLattice((Lx, Ly, Lz))

# Loop over superconducting order parameters.
for Δ0 in [0.03 * t]:
    # for d in ["e_z * p_x", "e_z * p_y", "e_z * (p_x + jp_y)", "(e_x + je_y) * (p_x + jp_y)"]:
    for d in ["e_x * p_x", "e_y * p_x", "e_x * p_y", "e_y * p_y"]:
        D = pwave(d)

        # Compare ferromagnetic and antiferromagnetic alignments
        # of the two magnetic layers for various separations δ.
        i10 = 20
        i11 = Ly // 2
        for δ in range(1, Lx - 2 * i10 + 1):
            i20 = i10 + δ
            i21 = i11

            # Construct the ferromagnetic Hamiltonian.
            system = Hamiltonian(lattice)
            with system as (H, Δ, V):
                for i in lattice.sites():
                    if i[0] == i10 and i[1] == i11:
                        H[i, i] = -μ * σ0 + M * σ3
                    elif i[0] == i20 and i[1] == i21:
                        H[i, i] = -μ * σ0 + M * σ3
                    else:
                        H[i, i] = -μ * σ0

                for i, j in lattice.bonds():
                    H[i, j] = -t * σ0
                    Δ[i, j] = -Δ0 * D(i, j)

            # Calculate the corresponding free energy.
            E_fm = free_energy(system)

            # Construct the antiferromagnetic Hamiltonian.
            with system as (H, Δ, V):
                for i in lattice.sites():
                    if i[0] == i20 and i[1] == i21:
                        H[i, i] = -μ * σ0 - M * σ3

            # Calculate the corresponding free energy.
            E_afm = free_energy(system)

            # Print the results in CSV format.
            print(f"{d}, {δ}, {E_fm - E_afm}")
