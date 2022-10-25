#!/usr/bin/env python

import csv
from time import time

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
from tqdm import tqdm, trange

from bodge import *

Lx = 64
Ly = 64

t = 1
μ = 0

Δ_0 = 0.1 * t

lattice = CubicLattice((Lx, Ly, 1))
system = Hamiltonian(lattice)


with system as (H, Δ, V):
    for i in lattice.sites():
        H[i, i] = -μ * σ0
        # Δ[i, i] = -Δ_0 * jσ2

    for i, j in lattice.bonds():
        H[i, j] = -t * σ0

A = spectral(system, [0.0], 1e-2 * Δ_0)[0]

D0 = np.zeros((Lx, Ly))
for i in lattice.sites():
    n = 4 * lattice[i]
    D0[i[0], i[1]] += np.real(A[n + 0, n + 0]) / (Lx * Ly)
    D0[i[0], i[1]] += np.real(A[n + 1, n + 1]) / (Lx * Ly)
D0 = np.mean(D0)

for desc in [
    "(e_x + je_y) * (p_x + jp_y)",
    "e_z * p_x",
    "e_z * p_y",
    "e_z * (p_x + jp_y)",
]:
    Δ_p = dvector(desc)

    with system as (H, Δ, V):
        for i in lattice.sites():
            H[i, i] = -μ * σ0
            # Δ[i, i] = -Δ_0 * jσ2

        for i, j in lattice.bonds():
            H[i, j] = -t * σ0
            Δ[i, j] = -Δ_0 * Δ_p(i, j)

    # H = system.scale * system.matrix.todense()
    # eigval, eigvec = eigh(H, subset_by_value=(0, 0.05 * Δ_0), driver="evr")
    # ε_0 = np.min(eigval)
    # D = np.zeros((Lx, Ly))
    # for n, ε_n in enumerate(eigval):
    #     if np.allclose(ε_n, ε_0):
    #         for i in lattice.sites():
    #             for r in range(4):
    #                 # print(eigvec[n, ...])
    #                 D[i[0], i[1]] += np.abs(eigvec[4 * lattice[i] + r, n]) ** 2

    # print(eigval)

    A = spectral(system, [0.0], 1e-2 * Δ_0)[0]

    Du = np.zeros((Lx, Ly))
    Dd = np.zeros((Lx, Ly))
    for i in lattice.sites():
        n = 4 * lattice[i]
        Du[i[0], i[1]] += np.real(A[n + 0, n + 0]) / (Lx * Ly)
        Dd[i[0], i[1]] += np.real(A[n + 1, n + 1]) / (Lx * Ly)

    Du /= D0
    Dd /= D0

    print(desc)
    plt.figure()
    plt.imshow(Du.T, vmin=0, origin="lower")
    plt.colorbar()
    plt.show()

    plt.figure()
    plt.imshow(Dd.T, vmin=0, origin="lower")
    plt.colorbar()
    plt.show()
    # bound.append(np.abs(eigvec[i]))

    # print(eigvec)
    # if np.allclose(ε_i, )
    # boundstate = [eigvec[i] for i ]

    # print(eigvals)
