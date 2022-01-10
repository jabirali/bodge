#!/usr/bin/env python

"""
This script is based on Sec. 2.6 of the Kwant documentation, where a simple 2D
system is tested using the BdG equations. It consists of a normal metal and a
superconductor with a barrier, and the differential conductance is calculated.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.linalg import eigh
from scipy.sparse import bsr_matrix
from scipy.sparse.linalg import inv
from tqdm import tqdm

from pybdg import *


# Parameters used in the example.
W = 10
L = 10
R = 1.5
Rpos = (3, 4)
μ = 1
Δ = 0.5
t = 1.0
m = 0.2

# Construct a system with the square lattice.
lat = Lattice((20, 20, 1))
sys = System(lat)

for i in lat.sites():
	sys.hopp[i][...] -= μ*σ0 + m*σ3
	sys.pair[i][...] += Δ*jσ2

for i, j in lat.neighbors():
	sys.hopp[i, j][...] -= t*σ0


# Obtain the Hamiltonian.
# H = bsr_matrix(sys.finalize(), blocksize=(4,4))
# I = bsr_matrix(np.identity(H.shape[0]))

# ε = np.arange(0, 3, 0.1)
# i = 3
# N = np.zeros_like(ε)
# for n, ε_n in tqdm(enumerate(ε)):
# 	G = (1j/np.pi) * inv((ε_n + 0.00001j) * I - H)
# 	N[n] = np.real(G[4*i, 4*i] + G[4*i+1, 4*i+1])/2

# plt.plot(ε, N)
# plt.show()

sys.finalize()
sys.diagonalize()
E, χ = sys.eigval, sys.eigvec

def delta(x):
	w = E.max()/100
	return np.exp(-x**2/(2*w**2)) / (w*np.sqrt(2*np.pi))

newval = np.arange(-E.max(), E.max(), E.max()/100)
dos = np.zeros_like(newval)
for n, E_n in enumerate(sys.eigval):
	X = (np.abs(χ[n, :, :, :])**2).sum(axis=-1).sum(axis=-1).mean() / 2
	for m, E_m in enumerate(newval):
		dos[m] += (delta(E_n - E_m) + delta(E_n + E_m)) * X

plt.plot(newval, dos)
# # sns.heatmap(dos, vmin=0, vmax=2)
plt.show()
# N = 1000
# dos = np.zeros((N, 10))
# e = np.arange(0, N)/100
# for i in range(N):
# 	for j in range(10):
# 		for n, E_n in enumerate(sys.eigval):
# 			dos[i,j] = delta(e[i] - E_n) * (np.abs(χ[n, lat[j,4,0], :, :])**2).sum()/2

# sns.heatmap(dos) # , vmin=0, vmax=1)
# plt.show()
# print(dos)
# for n, E_n in enumerate(sys.eigval):
	# print(E_n)
#
# X = np.zeros((W, L))
# for x, y, z in lat.sites():
# 	for n, E_n in enumerate(sys.eigval):
# 		X[x, y] += np.abs(sys.eigvec[n, lat[x, y, z], 0, 0])**2

# print(sys.eigval)
# sns.heatmap(X.T, vmin=0, vmax=1)
# plt.show()