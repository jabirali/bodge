#!/usr/bin/env python

import csv

import numpy as np
from tqdm import tqdm, trange

from bodge import *

Lx = 24
Ly = 24

t = 1
μ = 0
T = 1e-6 * t
U = 1

lattice = CubicLattice((Lx, Ly, 1))
system = Hamiltonian(lattice)
fermi = FermiMatrix(system, 1000)

with system as (H, Δ, V):
    for i in lattice.sites():
        H[i, i] = -μ * σ0
        V[i, i] = -U

    for i, j in lattice.bonds():
        H[i, j] = -t * σ0

with open("sweep3.csv", "w") as f:
    writer = csv.writer(f)

    for tol in tqdm([1e-1, 5e-2, 2e-2, 1e-2, 5e-3, 2e-3, 1e-3, 5e-4, 2e-4, 1e-4, None], desc="tol", leave=False):
        Δ_min = 0
        Δ_max = 1
        for n in trange(20, desc="Δ", leave=False):
            # Hamiltonian update.
            Δ_init = (Δ_min + Δ_max)/2
            with system as (H, Δ, V):
                for i in lattice.sites():
                    if V[i, i] != 0:
                        Δ[i, i] = Δ_init * jσ2

            # Convergence control.
            Δ2 = np.abs(fermi(T, tol).order_swave())
            Δ1 = np.where(Δ2 > 0, Δ_init, 0)

            if np.median(Δ2) > np.median(Δ1):
                Δ_min = Δ_init
            else:
                Δ_max = Δ_init

            writer.writerow([tol, (Δ_min + Δ_max)/2])
            f.flush()
