import numpy as np

from pybdg.lattice import *

lat = Lattice((1,1,3))
N = 3

# H = np.zeros((4 * N, 4 * N), dtype=np.complex64)
H = np.zeros((4 * N, 4 * N), dtype=np.float32)

# TODO: On-site.
for r in lat.sites():
	i = lat[r]

	Hii = H[4*i+0 : 4*i+2, 4*i+0 : 4*i+2]
	Hii[:, :] = 0.1

	Δii = H[4*i+0 : 4*i+2, 4*i+2 : 4*i+4]
	Δii[:,:] = 0.2


for r1, r2 in lat.neighbors():
	i, j = lat[r1], lat[r2]
	Hij = H[4*i+0 : 4*i+2, 4*j+0 : 4*j+2]
	Hij[:,:] = i+j
	# TODO: Gap.
	Δij = H[4*i+0 : 4*i+2, 4*j+2 : 4*j+4]
	Δij[:,:] = 2

for r1, r2 in lat.relevant():
	i, j = lat[r1], lat[r2]

	# Diagonal particle-hole symmetry.
	H[4*i+2 : 4*i+4, 4*j+2 : 4*j+4] = -H[4*i+0 : 4*i+2, 4*j+0 : 4*j+2].conj()

	# Anomalosu particle-hole symmetry.
	H[4*i+2 : 4*i+4, 4*j+0 : 4*j+2] = +H[4*i+0 : 4*i+2, 4*j+2 : 4*j+4].T.conj()

	# Neighbor hopping symmetry.   
	H[4*j : 4*j+4, 4*i : 4*i+4] = H[4*i : 4*i+4, 4*j : 4*j+4].T.conj()

print(H)