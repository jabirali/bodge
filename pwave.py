#!/usr/bin/env python

from time import time
import csv

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
from tqdm import tqdm, trange

from bodge import *
from bodge.typing import Coord

# TODO:
# - Test bound state by getting k=6 lowest positive eigenvalues.
# - Extract the eigenvectors as function of lattice coords: |u[lattice[i]]|^2 + |v[lattice[i]]^2 ?
# - Plot eigenvectors with eigenvalues below some threshold with different colors
# Alternatively, use Nagai's Ax=b solving of resolvent operator with sparse A.

Lx = 40
Ly = 24

t = 1
μ = 0.1
U = 1.5

T = 1e-6 * t

lattice = CubicLattice((Lx, Ly, 1))
system = Hamiltonian(lattice)
fermi = FermiMatrix(system, 2000)

D = np.zeros((Lx,Ly,3,3))

def dvector(desc: str):
    """Convert a d-vector expression into a p-wave gap function."""
    # Basis vectors for spin axes.
    e_x = np.array([[1], [0], [0]])
    e_y = np.array([[0], [1], [0]])
    e_z = np.array([[0], [0], [1]])

    je_x = 1j * e_x
    je_y = 1j * e_y
    je_z = 1j * e_z

    # Basis vectors for momentum.
    p_x = e_x.T
    p_y = e_y.T
    p_z = e_z.T

    jp_x = 1j * p_x
    jp_y = 1j * p_y
    jp_z = 1j * p_z

    # Convert the d-vector expression to a 3x3 numerical matrix.
    D = eval(desc)

    # Construct gap matrix Δ(p) = [d(p)⋅σ] jσ2 = [(D p) ⋅ σ] jσ2.
    # In practice, we do this by calculating Δ = Dᵀ σ jσ2, such
    # that we simply end up with the gap matrix Δ(p) = Δ ⋅ p.
    Δ = np.einsum('kp,kab,bc -> pac', D, σ, jσ2)

    # Function for evaluating Δ(p) on the lattice.
    def Δ_p(i: Coord, j: Coord):
        δ = np.array(j) - np.array(i)
        if norm(δ) == 1:
            return np.einsum('iab,i -> ab', Δ, δ)
        else:
            return 0 * σ0

    return Δ_p

Δ_p = dvector("e_z * p_y")
print(Δ_p((0,0,0), (1,0,0)))
print(Δ_p((0,0,0), (0,1,0)))
print(Δ_p((0,1,0), (0,0,0)))

Δ_p = dvector("p_y * e_z")
print(Δ_p((0,0,0), (1,0,0)))
print(Δ_p((0,0,0), (0,1,0)))
print(Δ_p((0,1,0), (0,0,0)))

with system as (H, Δ, V):
    for i in lattice.sites():
        if not (i[0] > 8 and i[0] < Lx-8 and i[1] > 8):
            H[i, i] = -μ * σ0

    for i, j in lattice.bonds():
        if not (i[0] > 8 and i[0] < Lx-8 and i[1] > 8) \
        and not (j[0] > 8 and j[0] < Lx-8 and j[1] > 8):
            H[i, j] = -t * σ0

tic = time()
print(system.matrix.tocsr().nnz)
F = fermi(T)
print(time() - tic)

with system as (H, Δ, V):
    for i in lattice.sites():
        H[i, i] = -μ * σ0

    for i, j in lattice.bonds():
        H[i, j] = -t * σ0

tic = time()
print(system.matrix.tocsr().nnz)
F = fermi(T)
print(time() - tic)

Lx = 24
Ly = 24

t = 1
μ = 0.1
U = 1.5

T = 1e-6 * t

lattice = CubicLattice((Lx, Ly, 1))
system = Hamiltonian(lattice)
fermi = FermiMatrix(system, 2000)

tic = time()
print(system.matrix.tocsr().nnz)
F = fermi(T)
print(time() - tic)
