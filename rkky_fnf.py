#!/usr/bin/env python

import numpy as np

from bodge import *
from bodge.utils import ldos

t = 1
μ = 0.5
δ = 10

for Ln in range(20):
    Lm = 5

    Lx = Ln + 2 * Lm
    Ly = 30


    M = 2 * t

    lattice = CubicLattice((Lx, Ly, 1))
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

    F1 = free_energy(system)

    with system as (H, Δ, V):
        for i in lattice.sites():
            if i[0] < Lm:
                H[i, i] = -μ * σ0 - M * σ3

    F2 = free_energy(system)

    print(Ln, F1 - F2)