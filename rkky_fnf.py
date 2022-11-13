#!/usr/bin/env python

import numpy as np

from bodge import *
from bodge.utils import ldos

t = 1
μ = 0.5

for δ in range(1, 101):
    Lx = 24
    Ly = 16
    Lz = 16

    M = t/2

    lattice = CubicLattice((Lx, Ly, Lz))
    system = Hamiltonian(lattice)


    with system as (H, Δ, V):
        for i in lattice.sites():
            if i[0] == 0 or i[0] == 1:
                H[i, i] = -μ * σ0 + M * σ3
            elif i[0] == 2 + δ or i[0] == 3 + δ:
                H[i, i] = -μ * σ0 + M * σ3
            else:
                H[i, i] = -μ * σ0

        for i, j in lattice.bonds():
            H[i, j] = -t * σ0

    F_fm = free_energy(system)

    with system as (H, Δ, V):
        for i in lattice.sites():
            if i[0] == 2 + δ or i[0] == 3 + δ:
                H[i, i] = -μ * σ0 - M * σ3

    F_afm = free_energy(system)

    print(f"{δ}, {(F_fm - F_afm)/(Ly*Lz)}")