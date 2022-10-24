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

Lx = 24
Ly = 24

t = 1
μ = 0

T = 1e-6 * t

lattice = CubicLattice((Lx, Ly, 1))
system = Hamiltonian(lattice)

Δ_0 = t
Δ_p = dvector("e_z * p_y")

with system as (H, Δ, V):
    for i in lattice.sites():
        H[i, i] = -μ * σ0

    for i, j in lattice.bonds():
        H[i, j] = -t * σ0
        Δ[i, j] = -Δ_0 * Δ_p(i, j)

H = system.compile()

eigval, eigvec = eigsh(system.scale * H, 8, which="SM")
# print(eigvec.shape)
eigvec = eigvec.T.reshape((eigval.size, -1, 4))

# eigval, eigvec = system.diagonalize()

# bound = []
minim = np.min(np.abs(eigval))

D = np.zeros((Lx, Ly))
for n, ε_n in enumerate(eigval):
    if ε_n > 0 and np.allclose(ε_n, minim):
        for i in lattice.sites():
            # print(i, lattice[i])
            for r in range(4):
                D[i[0], i[1]] += np.abs(eigvec[n, lattice[i], r]) ** 2
plt.imshow(D)
plt.show()
# bound.append(np.abs(eigvec[i]))

# print(eigvec)
# if np.allclose(ε_i, )
# boundstate = [eigvec[i] for i ]

# print(eigvals)
