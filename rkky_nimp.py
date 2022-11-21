#!/usr/bin/env python

"""RKKY interactions between two magnetic impurities."""

import numpy as np

from bodge import *
from bodge.common import *
from bodge.utils import ldos

# Tight-binding parameters.
t = 1
μ = -t
M = +t/2

# Construct a 2D lattice.
Lx = 100
Ly = 100
Lz = 1

lattice = CubicLattice((Lx, Ly, Lz))

# Prepare impurity coordinates.
i10 = 20
i11 = Ly // 2

i20 = i10 + 1
i21 = i11

# Loop over superconducting order parameters.
for Δ0 in [0.00, 0.01*t, 0.03*t, 0.10*t, 0.30*t, 1.0*t]:
        print(f"Δ0 = {Δ0}")

        # Compare ferromagnetic and antiferromagnetic alignments
        # of the two magnetic layers for various separations δ.
        for δ in range(1, Ly - 2*Le):
            # Construct the ferromagnetic Hamiltonian.
            system = Hamiltonian(lattice)
            with system as (H, Δ, V):
                for i in lattice.sites():
                    Δ[i, i] = -Δ0 * jσ2
                    if i[0] == i10 and i[1] == i11:
                        H[i, i] = -μ * σ0 + M * σ3
                    elif i[0] == i20 and i[1] == i21:
                        H[i, i] = -μ * σ0 + M * σ3
                    else:
                        H[i, i] = -μ * σ0

                for i, j in lattice.bonds()
                H[i, j] = -t * σ0

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
            print(f"{δ}, {E_fm - E_afm}")
