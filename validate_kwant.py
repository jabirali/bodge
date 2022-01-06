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
		sys.hopp[x, y, z] += R * σ0

for r1, r2 in lat.neighbors():
	sys.hopp[r1, r2] = -t * σ0

# Obtain the Hamiltonian.
sys.finalize()
sys.diagonalize()


X = np.zeros((W, L))
for x, y, z in lat.sites():
	X[x, y] += np.abs(sys.eigvec[:, lat[x, y, z], 0, 0].sum())

print(sys.eigval)
sns.heatmap(X.T, vmin=0, vmax=1)
plt.show()