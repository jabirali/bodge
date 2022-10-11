#!/usr/bin/env python

import csv
from time import time

import numpy as np
from tqdm import tqdm, trange

from bodge import *

Lx = 24
Ly = 200

t = 1
μ = 0
T = 2e-3 * t
U = 1
Δ0 = 0.06643533706665039

lattice = CubicLattice((Lx, Ly, 1))
system = Hamiltonian(lattice)
fermi = FermiMatrix(system, 600)

with system as (H, Δ, V):
    for i in lattice.sites():
        H[i, i] = -μ * σ0
        # if i[0] <= 15:
        Δ[i, i] = Δ0 * jσ2
        V[i, i] = -U

    for i, j in lattice.bonds():
        H[i, j] = -t * σ0

with open("sweep4.csv", "w") as f:
    writer = csv.writer(f)

    for tol in tqdm([1e-0, 3e-1, 1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 0], desc="tol", leave=False):
        start = time()
        Δ = np.abs(fermi(T, tol).order_swave())
        end = time()

        writer.writerow([tol, np.mean(Δ), end - start])
        f.flush()
