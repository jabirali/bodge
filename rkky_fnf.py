#!/usr/bin/env python

import numpy as np

from bodge import *
from bodge.utils import ldos

t = 1
μ = 0.5

Lm = 2
for δ in range(1, 101):
    Lx = 2 * Lm + δ
    Ly = 20
    Lz = 20

    M = t/2

    lattice = CubicLattice((Lx, Ly, Lz))
    system = Hamiltonian(lattice)


    with system as (H, Δ, V):
        for i in lattice.sites():
            if i[0] < Lm:
                H[i, i] = -μ * σ0 + M * σ3
            elif i[0] > (Lx - 1) - Lm:
                H[i, i] = -μ * σ0 + M * σ3
            else:
                H[i, i] = -μ * σ0

        for i, j in lattice.bonds():
            H[i, j] = -t * σ0

        # for i in lattice.sites():
        #     if i[1] == 0:
        #         H[i, (i[0], i[1], )]
        # H[0, Lx-1] = -t * σ0
        # H[Lx-1, 0] = -t * σ0
        # H[0, Lx-1] = -t * σ0
        # H[Lx-1, 0] = -t * σ0

    F_fm = free_energy(system)

    with system as (H, Δ, V):
        for i in lattice.sites():
            if i[0] < Lm:
                H[i, i] = -μ * σ0 - M * σ3

    F_afm = free_energy(system)

    print(f"{δ}, {(F_fm - F_afm)/(Ly*Lz)}")