#!/usr/bin/env python

# Minimal validation example based on Sec. 2.6 of the Kwant documentation.

import numpy as np
import matplotlib.pyplot as plt

from scipy.linalg import eigh

from pybdg import *


# Parameters used in the example.
W = 10
L = 10
R = 1.5
Rpos = (3, 4)
μ = 0.4
Δ = 0.1
Δpos = 4
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
		sys.hopp[x, y, z] += R

for r1, r2 in lat.neighbors():
	sys.hopp[r1, r2] = -t * σ0

# Obtain the Hamiltonian.
H = sys.asarray()

E, X = eigh(H, driver='evr', subset_by_value=(0, np.inf))
χ = X.T.reshape((E.size, -1, 2, 2))  # Indices: n, i, eh, ↑↓

# print(np.sum(np.abs(χ[1,:,1,0])))
X = np.zeros((W, L))
for x, y, z in lat.sites():
	X[x, y] += np.abs(χ[:, lat[x, y, z], 0, :]).sum(axis=0).sum(axis=-1)

plt.imshow(X.T)
plt.show()