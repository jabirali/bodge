#!/usr/bin/env python

"""
This is a test script that constructs a simple tight-binding Hamiltonian for
a superconducting system and subsequently calculates the density of states.
"""
import enum
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# from scipy.linalg import eigh
# from scipy.sparse import bsr_matrix, csc_matrix
# from scipy.sparse.linalg import inv
from scipy.linalg import inv
from tqdm import tqdm

from pybdg import *


t = 1.0
μ = t/2
Δ0 = t/2
m3 = t/5

lattice = Cubic((10, 10, 20))
system = System(lattice)

with system as (H, Δ):
	for i in lattice.sites():
		x, y, z = i
		H[i, i] = -μ * σ0 #- m3 * σ3

		if x >= 5:
			Δ[i, i] = Δ0 * jσ2

	for i, j in lattice.neighbors():
		H[i, j] = -t * σ0

system.diagonalize()

# Plot the DOS.
E, χ = system.eigval, system.eigvec

def delta(x):
	w = E.max()/75
	return np.exp(-x**2/(2*w**2)) / (w*np.sqrt(2*np.pi))

newval = np.arange(-2.0, 2.0, 2.0/100)
dos = np.zeros((10, 200))
for n, E_n in enumerate(system.eigval):
	X = np.zeros((10,))
	for i in lattice.sites():
		x, y, z = i
		i = lattice[i]

		X[x] += (np.abs(χ[n, i, :])**2).sum(axis=-1) / 2
	for m, E_m in enumerate(newval):
		for x in range(10):
			dos[x, m] += (delta(E_n - E_m) + delta(E_n + E_m)) * X[x]

sns.heatmap(dos)
plt.show()

# plt.plot(newval, dos)
# plt.show()

# Sparse matrices.
# H = bsr_matrix(system.data, blocksize=(4,4))
# I = bsr_matrix(np.identity(H.shape[0]))

# H = csc_matrix(system.data)
# I = csc_matrix(np.identity(system.data.shape[0]))

# # Green function solution.
# H = system.data
# I = np.identity(H.shape[0])

# ε = np.arange(0, 3, 0.01)
# i = 3
# N = np.zeros_like(ε)
# for n, ε_n in tqdm(enumerate(ε)):
# 	G = (1j/np.pi) * inv((ε_n + 0.05j) * I - H)
# 	N[n] = np.real(G[4*i, 4*i] + G[4*i+1, 4*i+1])

# E = np.hstack([-ε[::-1], ε])
# N = np.hstack([N[::-1], N])
# plt.plot(E, N)
# plt.show()

