#!/usr/bin/env python

import csv

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
from tqdm import tqdm, trange

from bodge import *
from bodge.typing import Coord

Lx = 24
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
    e_x = np.array([[1, 0, 0]])
    e_y = np.array([[0, 1, 0]])
    e_z = np.array([[0, 0, 1]])

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

    # Construct gap matrix Δ(p) = [d(p)⋅σ] jσ2 = p ⋅ (D σ jσ2)
    # for p along the three cardinal directions {e_x, e_y, e_z}.
    Δ = np.einsum('pq,qab,bc -> pac', D, σ, jσ2)

    # Function for evaluating Δ(p) on the lattice.
    def Δ_p(i: Coord, j: Coord):
        δ = np.array(j) - np.array(i)
        if norm(δ) == 1:
            return np.einsum('i,iab -> ab', δ, Δ)
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

# with system as (H, Δ, V):
#     for i in lattice.sites():
#         H[i, i] = -μ * σ0

#     for i, j in lattice.bonds(axis=1):
#         H[i, j] = -t * σ0
#         Δ[i, j] = -Δ_p(i, j)
#         print(Δ_p(i,j))
