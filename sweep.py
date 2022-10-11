#!/usr/bin/env python

import csv
from time import time

import numpy as np
from tqdm import tqdm, trange

from bodge import *

with open("sweep5.csv", "w") as f:
    writer = csv.writer(f)
    for L in tqdm([64, 128, 258, 512, 1024, 2048, 4096]):
        Lx = L
        Ly = 8

        t = 1
        μ = 0
        T = 1e-6 * t
        U = 1
        Δ0 = 0.06643533706665039

        tol = 0

        lattice = CubicLattice((Lx, Ly, 1))
        system = Hamiltonian(lattice)
        fermi = FermiMatrix(system, 1000)

        with system as (H, Δ, V):
            for i in lattice.sites():
                H[i, i] = -μ * σ0
                # if i[0] <= 15:
                Δ[i, i] = Δ0 * jσ2
                V[i, i] = -U

            for i, j in lattice.bonds():
                H[i, j] = -t * σ0

        start = time()
        Δ = np.abs(fermi(T, tol).order_swave())
        end = time()

        writer.writerow([Lx * Ly, np.mean(Δ), end - start])
        f.flush()
