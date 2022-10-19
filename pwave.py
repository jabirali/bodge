#!/usr/bin/env python

import csv

import matplotlib.pyplot as plt
import numpy as np
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

    # Convert the expression to numerics.
    D = eval(desc)  # TODO: Make this safe again.
    σ = np.stack([σ1 @ jσ2, σ2 @ jσ2, σ3 @ jσ2])

    # Construct the actual dvector in matrix form
    d = np.einsum('ij,jnm->inm', D, σ)

    # Function for evaluating the dvector.
    def deval(i: Coord, j: Coord):
        diff = Coord(np.array(j) - np.array(i))

        match diff:
            case (1, 0, 0):
                return d[0]
            case (0, 1, 0):
                return d[1]
            case (0, 0, 1):
                return d[2]
            case _:
                return 0*σ0

    return deval

Δ_p = dvector("0.125 * (e_x + je_y) * (p_x + jp_y)")

with system as (H, Δ, V):
    for i in lattice.sites():
        H[i, i] = -μ * σ0

    for i, j in lattice.bonds():
        H[i, j] = -t * σ0
        Δ[i, j] = -Δ_p(i, j)
