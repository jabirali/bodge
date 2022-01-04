import numpy as np

from pybdg.lattice import *

class System:
	def __init__(self, lat):
		# Lattice instance used as coordinates for the system.
		self.lat = lat

		# Number of lattice points that the system has in total.
		self.dim = np.prod(lat.dims)

		# Hamiltonian matrix used in the tight-binding treatment. This is
		# a 4N×4N matrix, where we assume particle-hole and spin degrees
		# of freedom, and the remaining N is the number of lattice points.
		self.mat = np.zeros((4*self.dim, 4*self.dim), dtype=np.float32)

		# Simplified accessors for the diagonal and anomalous matrix parts.
		# Only these need to be explicitly defined, since the class is smart
		# enough to construct the remaining terms via symmetry considerations.
		self.ham = {}
		self.gap = {}
		for r_i, r_j in self.lat.relevant():
			# Convert indices from coordinate form.
			i, j = self.lat[r_i], self.lat[r_j]

			# Two-index notation for every interaction.
			self.ham[r_i, r_j] = self.mat[4*i+0 : 4*i+2, 4*j+0 : 4*j+2]
			self.gap[r_i, r_j] = self.mat[4*i+0 : 4*i+2, 4*j+2 : 4*j+4]

			# One-index notation for on-site interactions.
			if r_i == r_j:
				self.ham[r_i] = self.ham[r_j, r_j]
				self.gap[r_i] = self.gap[r_j, r_j]

	def finalize(self):
		for r_i, r_j in self.lat.relevant():
			# Convert indices from coordinate form.
			i, j = self.lat[r_i], self.lat[r_j]

			# Diagonal particle-hole symmetry.
			self.mat[4*i+2 : 4*i+4, 4*j+2 : 4*j+4] = \
				-self.mat[4*i+0 : 4*i+2, 4*j+0 : 4*j+2].conj()

			# Anomalous particle-hole symmetry.
			self.mat[4*i+2 : 4*i+4, 4*j+0 : 4*j+2] = \
				+self.mat[4*i+0 : 4*i+2, 4*j+2 : 4*j+4].T.conj()

			# Neighbor hopping symmetry.   
			self.mat[4*j+0 : 4*j+4, 4*i+0 : 4*i+4] = \
				+self.mat[4*i+0 : 4*i+4, 4*j+0 : 4*j+4].T.conj()

	def asmatrix(self):
		self.finalize()
		return self.mat

lat = Lattice((1,1,3))
sys = System(lat)

for r in lat.sites():
	sys.ham[r][:, :] = 0.1 + sum(r)
	sys.gap[r][:, :] = 0.2

for r1, r2 in lat.neighbors():
	sys.ham[r1, r2][:, :] = sum(r1)+sum(r2)
	sys.gap[r1, r2][:, :] = 2

print(sys.mat)

sys.finalize()
print(sys.mat)
# N = 3

# # H = np.zeros((4 * N, 4 * N), dtype=np.complex64)
# H = np.zeros((4 * N, 4 * N), dtype=np.float32)

# # TODO: On-site.
# for r in lat.sites():
# 	i = lat[r]

# 	Hii = H[4*i+0 : 4*i+2, 4*i+0 : 4*i+2]
# 	Hii[:, :] = 0.1

# 	Δii = H[4*i+0 : 4*i+2, 4*i+2 : 4*i+4]
# 	Δii[:,:] = 0.2


# for r1, r2 in lat.neighbors():
# 	i, j = lat[r1], lat[r2]
# 	Hij = H[4*i+0 : 4*i+2, 4*j+0 : 4*j+2]
# 	Hij[:,:] = i+j
# 	# TODO: Gap.
# 	Δij = H[4*i+0 : 4*i+2, 4*j+2 : 4*j+4]
# 	Δij[:,:] = 2

# for r1, r2 in lat.relevant():
# 	i, j = lat[r1], lat[r2]

# 	# Diagonal particle-hole symmetry.
# 	H[4*i+2 : 4*i+4, 4*j+2 : 4*j+4] = -H[4*i+0 : 4*i+2, 4*j+0 : 4*j+2].conj()

# 	# Anomalosu particle-hole symmetry.
# 	H[4*i+2 : 4*i+4, 4*j+0 : 4*j+2] = +H[4*i+0 : 4*i+2, 4*j+2 : 4*j+4].T.conj()

# 	# Neighbor hopping symmetry.   
# 	H[4*j : 4*j+4, 4*i : 4*i+4] = H[4*i : 4*i+4, 4*j : 4*j+4].T.conj()

# print(H)