#!/usr/bin/env python

import numpy as np

from bodge import *
from bodge.utils import ldos

Lx = 101
Ly = 101

t = 1
μ = 0.5
δ = 10

Δ_0 = 0.1 * t
S_0 = 2 * t

lattice = CubicLattice((Lx, Ly, 1))
system = Hamiltonian(lattice)


def spin(desc: str):
    e_x = np.array([1, 0, 0])
    e_y = np.array([0, 1, 0])
    e_z = np.array([0, 0, 1])

    S = eval(desc)

    return np.einsum("s,sab -> ab", S, σ)


with system as (H, Δ, V):
    for i in lattice.sites():
        Δ[i, i] = -Δ_0 * jσ2

        if i[0] == (Lx - δ) // 2 and i[1] == Ly // 2:
            H[i, i] = -μ * σ0 - S_0 * spin("+e_z")
        elif i[0] == (Lx + δ) // 2 and i[1] == Ly // 2:
            H[i, i] = -μ * σ0 - S_0 * spin("-e_z")
        else:
            H[i, i] = -μ * σ0

    for i, j in lattice.bonds():
        H[i, j] = -t * σ0

sites = [i for i in lattice.sites() if i[1] == Ly // 2]
energies = np.linspace(0, 2 * Δ_0, 51)

df = ldos(system, sites, energies)
df.to_csv("rkky.csv")
