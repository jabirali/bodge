#!/usr/bin/env python

import csv
from time import time

import numpy as np
from tqdm import tqdm, trange

from bodge import *

with open("sweep5.csv", "w") as f:
    writer = csv.writer(f)
    for sites in tqdm([32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 2*16384, 4*16384, 8*16384]):
        Lx = sites
        Ly = 1

        t = 1
        μ = 0
        T = 1e-6 * t
        U = 1
        Δ0 = 0.06643533706665039

        tol = 1e-3

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


        start = time()
        Δ = np.abs(fermi(T, tol).order_swave())
        end = time()

        writer.writerow([Lx * Ly, np.mean(Δ), end - start])
        f.flush()
