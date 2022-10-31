#!/usr/bin/env python


import csv
from time import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from bodge import *
from bodge.utils import ldos

Lx = 320
Ly = 320

t = 1
μ = 0.5

for Δ_0 in [1.0, 0.7, 0.3, 0.1, 0.07, 0.03, 0.01]:
    lattice = CubicLattice((Lx, Ly, 1))
    system = Hamiltonian(lattice)

    d = dvector("e_x * p_y - e_y * p_x")

    with system as (H, Δ, V):
        for i in lattice.sites():
            H[i, i] = -μ * σ0

        for i, j in lattice.bonds():
            H[i, j] = -t * σ0
            Δ[i, j] = -Δ_0 * d(i, j)

    sites = [
        # (0, 0, 0),
        (Lx // 2, Ly // 2, 0),
    ]

    df = ldos(system, sites, [0], 0.05 * Δ_0)
    # corner = np.array(df[(df.y == 0) & (df.ε == +0.0)].dos)[0]
    edge = np.array(df[(df.y > 0) & (df.ε == +0.0)].dos)[0]
    print(f"{Δ_0},{edge}")
