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

R = None

lattice = CubicLattice((Lx, Ly, 1))
system = Hamiltonian(lattice)

with system as (H, Δ, V):
    for i in lattice.sites():
        H[i, i] = -μ * σ0

    for i, j in lattice.bonds():
        H[i, j] = -t * σ0

with open("sweep.csv", "w") as f:
    writer = csv.writer(f)

    for N in tqdm([10, 50, 100, 200, 300, 400, 500, 600, 800, 1000], desc="N", leave=False):
        F = FermiMatrix(system, N)

        for U in tqdm(np.linspace(0, 2, 21), desc="U", leave=False):
            with system as (H, Δ, V):
                for i in lattice.sites():
                    V[i, i] = -U

            Δ_min = 0
            Δ_max = 2
            for n in trange(20, desc="Δ", leave=False):
                # Hamiltonian update.
                Δ_init = (Δ_min + Δ_max)/2
                with system as (H, Δ, V):
                    for i in lattice.sites():
                        if V[i, i] != 0:
                            Δ[i, i] = Δ_init * jσ2

                # Convergence control.
                Δ2 = np.abs(F(T).order_swave())
                Δ1 = np.where(Δ2 > 0, Δ_init, 0)

                if np.mean(Δ2) > np.mean(Δ1):
                    Δ_min = Δ_init
                else:
                    Δ_max = Δ_init

            writer.writerow([N, U, (Δ_min + Δ_max)/2])
            f.flush()
