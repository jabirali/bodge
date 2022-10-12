#!/usr/bin/env python

import csv
from time import time

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm, trange

from bodge import *

with open("sweep5.csv", "w") as f:
    writer = csv.writer(f)

    Lx = 32
    Ly = 32

    t = 1
    μ = 0
    T = 1e-3 * t
    U = 1
    Δ0 = 0.06643533706665039

    lattice = CubicLattice((Lx, Ly, 1))
    system = Hamiltonian(lattice)
    fermi = FermiMatrix(system, 400)

    with system as (H, Δ, V):
        for i in lattice.sites():
            H[i, i] = -μ * σ0
            if i[0] <= 127:
                Δ[i, i] = Δ0 * jσ2
                V[i, i] = -U

        for i, j in lattice.bonds():
            H[i, j] = -t * σ0

    start = time()
    Δ = np.abs(fermi(T).order_swave())
    end = time()

    # print()
    # plt.figure()
    # plt.imshow(Δ, vmin=0)
    # plt.colorbar()
    # plt.show()
    # print()

    writer.writerow([Δ[127, 15, 0], end - start])
    print([Δ[127, 15, 0], end - start])
    f.flush()
