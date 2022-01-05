#!/usr/bin/env python

# Minimal validation example based on Sec. 2.6 of the Kwant documentation.

import numpy as np

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
lat = Lattice((W, L, 10))
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
N = 1000
χ = X.T.reshape((2*N, N, 2, 2))  # Indices: n, i, eh, ↑↓
print(χ[:, -1, 0, 0])

# print(E.shape)  # 4000/2 positive eigenvalues
# # print(X.shape)
# # print(E)
# χ = [X[:,n] for n, E_n in enumerate(E) if E_n > 0]
# print(len(χ))  # 4000/2 corresponding eigenvectors

# print(χ[0].shape)  # 2 particles * 2 spins * 1000 sites 

# u_up = np.array([χ_n[0] for χ_n in χ])
# u_dn = np.array([χ_n[1] for χ_n in χ])
# v_up = np.array([χ_n[2] for χ_n in χ])
# v_dn = np.array([χ_n[3] for χ_n in χ])

# print(len(u_up))
# # print(χ[0].shape	)

# # print(χ[100])
# # print(len(χ[100]))

# # u_up = [χ_n[0] for χ_n in χ]
# # print(u_up)

# # E = [E_n for E_n in E if E_n > 0]
# # print(E)
# # # print(χ)