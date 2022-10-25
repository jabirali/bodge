#!/usr/bin/env python

import csv
from time import time

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
from scipy.sparse.linalg import eigsh
from tqdm import tqdm, trange

from bodge import *

# TODO:
# - Test bound state by getting k=6 lowest positive eigenvalues.
# - Extract the eigenvectors as function of lattice coords: |u[lattice[i]]|^2 + |v[lattice[i]]^2 ?
# - Plot eigenvectors with eigenvalues below some threshold with different colors
# Alternatively, use Nagai's Ax=b solving of resolvent operator with sparse A.

Lx = 32
Ly = 32

t = 1
μ = 0

T = 1e-6 * t

lattice = CubicLattice((Lx, Ly, 1))
system = Hamiltonian(lattice)

for desc in [
    "e_x * p_x",
    "e_x * p_y",
    "e_x * (p_x + ip_y)",
]:
    Δ_0 = 0.1 * t
    Δ_p = dvector(desc)

    with system as (H, Δ, V):
        for i in lattice.sites():
            H[i, i] = -μ * σ0
            # Δ[i, i] = -Δ_0 * jσ2

        for i, j in lattice.bonds():
            H[i, j] = -t * σ0
            Δ[i, j] = -Δ_0 * Δ_p(i, j)

    A = spectral(system, [0.0], 1e-5)[0]

    D = np.zeros((Lx, Ly))
    for i in lattice.sites():
        for r in range(4):
            n = 4 * lattice[i]
            D[i[0], i[1]] += np.real(A[n + 0, n + 0] + A[n + 1, n + 1]) / (Lx * Ly)

    print(desc)
    plt.imshow(D.T, vmin=0, origin="lower")
    plt.xticks([0, 16, 32, 48, 64])
    plt.yticks([0, 16, 32, 48, 64])
    plt.colorbar()
    plt.show()
    # bound.append(np.abs(eigvec[i]))

    # print(eigvec)
    # if np.allclose(ε_i, )
    # boundstate = [eigvec[i] for i ]

    # print(eigvals)
