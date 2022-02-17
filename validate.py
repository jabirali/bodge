#!/usr/bin/env python

"""
This is a test script that constructs a simple tight-binding Hamiltonian for
a superconducting system and subsequently calculates the density of states.
"""
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from tqdm import tqdm

from pybdg import *


t = 1.0
μ = +3*t
Δ0 = t/2
m3 = t/5

lattice = Cubic((100, 10, 1))
system = System(lattice)
with system as (H, Δ):
	for i in lattice.sites():
		H[i, i] = -μ * σ0 - m3 * σ3
		Δ[i, i] = Δ0 * jσ2

	for i, j in lattice.neighbors():
		H[i, j] = -t * σ0

	# for i, j in lattice.neighbors(axis=1):
	# 	H[i, j] = -1e-10 * σ0

system.chebyshev()

# I = system.identity
# H = system.hamiltonian
# G = H @ I
# print(I.nnz)
# print(H.nnz)
# print(G.nnz)

# G.eliminate_zeros()
# print(G.nnz)

# G.data[np.abs(G.data) < 1e-8] = 0
# G.eliminate_zeros()
# G.eliminate_zeros()
