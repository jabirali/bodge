#!/usr/bin/env python

"""
This is a test script that constructs a simple tight-binding Hamiltonian for
a superconducting system and subsequently calculates the density of states.
"""
import enum
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numba

# from scipy.linalg import eigh
# from scipy.sparse import bsr_matrix, csc_matrix
# from scipy.sparse.linalg import inv
from scipy.linalg import inv
from tqdm import tqdm

from pybdg import *


t = 1.0
μ = +3*t
Δ0 = t/3
# m3 = t/5

lattice = Cubic((10, 10, 15))
system = System(lattice)

with system as (H, Δ):
	for i in lattice.sites():
		H[i, i] = -μ * σ0 #- m3 * σ3
		Δ[i, i] = Δ0 * jσ2

	for i, j in lattice.neighbors():
		H[i, j] = -t * σ0

system.diagonalize()

# Plot the DOS.
# E, χ = system.eigval, system.eigvec

@numba.njit()
def delta(x):
	w = 0.05
	return np.exp(-x**2/(2*w**2)) / (w*np.sqrt(2*np.pi))

N = 300
newval = np.arange(-3.0, 3.0, 6.0/N)
dos = np.zeros((N,))
for n, E_n in enumerate(system.eigval):
	e = (np.abs(system.eigvec[n, :, 0:1])**2).sum(axis=-1).mean() / 2
	h = (np.abs(system.eigvec[n, :, 2:3])**2).sum(axis=-1).mean() / 2
	for m, E_m in enumerate(newval):
		dos[m] += (delta(E_n - E_m)) * e + (delta(E_n + E_m)) * h

plt.plot(newval, dos)
plt.xlabel("Energy ε")
plt.ylabel("Density of states N(ε)")
plt.axvline(-μ/2, 0, 1, color='k')
plt.show()

# dos = np.zeros((10, 200))
# for n, E_n in enumerate(system.eigval):
# 	X = np.zeros((10,))
# 	for i in lattice.sites():
# 		x, y, z = i
# 		i = lattice[i]

# 		X[x] += (np.abs(χ[n, i, :])**2).sum(axis=-1) / 2
# 	for m, E_m in enumerate(newval):
# 		for x in range(10):
# 			dos[x, m] += (delta(E_n - E_m) + delta(E_n + E_m)) * X[x]

# sns.heatmap(dos)
# plt.show()

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

