#!/usr/bin/env python

import csv
from time import time

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
from tqdm import tqdm, trange

from bodge import *

# TODO:
# - Test bound state by getting k=6 lowest positive eigenvalues.
# - Extract the eigenvectors as function of lattice coords: |u[lattice[i]]|^2 + |v[lattice[i]]^2 ?
# - Plot eigenvectors with eigenvalues below some threshold with different colors
# Alternatively, use Nagai's Ax=b solving of resolvent operator with sparse A.

Lx = 20
Ly = 20

t = 1
μ = 0.1

T = 1e-6 * t

lattice = CubicLattice((Lx, Ly, 1))
system = Hamiltonian(lattice)
# fermi = FermiMatrix(system, 2000)

Δ_p = dvector("e_z * p_y")
print(Δ_p((0, 0, 0), (1, 0, 0)))
print(Δ_p((0, 0, 0), (0, 1, 0)))
print(Δ_p((0, 1, 0), (0, 0, 0)))

Δ_p = dvector("p_y * e_z")
print(Δ_p((0, 0, 0), (1, 0, 0)))
print(Δ_p((0, 0, 0), (0, 1, 0)))
print(Δ_p((0, 1, 0), (0, 0, 0)))
