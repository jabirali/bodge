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

from pybdg import *


# Parameters used in the example.
W = 10
L = 10
R = 1.5
Rpos = (3, 4)
μ = 0.4
Δ = 1
Δpos = 6
t = 1.0

# Construct a system with the square lattice.
lat = Lattice((W, L, 1))
sys = System(lat)

for x, y, z in lat.sites():
	# At every site, we add a diagonal term 4t-μ.
	sys.hopp[x, y, z] += (4*t - μ) * σ0

	# Superconductivity is included only for x>Δpos.
	if x > Δpos:
		sys.pair[x, y, z] += Δ * jσ2

	# Barrier is located between Rpos coordinates.
	if x >= Rpos[0] and x < Rpos[1]:
		sys.hopp[x, y, z] += R * σ0

for r1, r2 in lat.neighbors():
	sys.hopp[r1, r2] = -t * σ0

# Obtain the Hamiltonian.
sys.finalize()
sys.diagonalize()

E, χ = sys.eigval, sys.eigvec

def delta(x):
	w = 0.05
	return np.exp(-x**2/(2*w**2)) / (w*np.sqrt(2*np.pi))

newval = np.arange(0, 6, 0.01)
dos = np.zeros((len(newval), W))
for n, E_n in enumerate(sys.eigval):
	for m, E_m in enumerate(newval):
		for j in range(W):
			X = (np.abs(χ[n, lat[j, 5, 0], :, :])**2).sum()
			D = delta(E_n/2 - E_m)

			if X > 0 and D > 0:
				dos[m, j] += D * X

sns.heatmap(dos, vmin=0)
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