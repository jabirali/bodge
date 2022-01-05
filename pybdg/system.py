import numpy as np
from scipy.sparse import lil_matrix


# Define the Pauli matrices used to represent spin.
σᵒ = np.array([[+1,  0 ], [ 0, +1]], dtype=np.complex64)
σˣ = np.array([[ 0, +1 ], [+1,  0]], dtype=np.complex64)
σʸ = np.array([[ 0, -1j], [+1j, 0]], dtype=np.complex64)
σᶻ = np.array([[+1,  0 ], [ 0, -1]], dtype=np.complex64)


class System:
	"""Representation of a physical system in the tight-binding limit.

	This can be used to construct Hamiltonian matrices for condensed matter
	systems that exhibit particle-hole and spin degrees of freedom. Instead
	of explicitly constructing the whole matrix, this class allows you to
	specify the minimum number of matrix elements required as the rest are
	autofilled via symmetries. Moreover, it allows you to use `Lattice`
	coordinates instead of matrix indices to fill out these elements.
	"""
	def __init__(self, lattice):
		# Lattice instance used as basis coordinates for the system.
		self.lattice = lattice

		# Number of lattice points that the system has. The integers in front of
		# the lattice size are the local degrees of freedom at each lattice site.
		self.shape = (4*lattice.size, 4*lattice.size)

		# Hamiltonian matrix used in the tight-binding treatment.
		self.data = np.zeros(self.shape, dtype=np.complex64)

		# Convenience accessors for electron-electron and electron-hole blocks
		# in the Hamiltonian matrix at each site. Note that the hole-electron
		# and hole-hole blocks are autofilled using the Hamiltonian symmetries.
		self.site = {}
		self.hopp = {}
		self.pair = {}
		for r_i, r_j in self.lattice.relevant():
			# Convert indices from coordinate form.
			i, j = self.lattice[r_i], self.lattice[r_j]

			# Two-index notation for every interaction.
			self.hopp[r_i, r_j] = self.data[4*i+0 : 4*i+2, 4*j+0 : 4*j+2]
			self.pair[r_i, r_j] = self.data[4*i+0 : 4*i+2, 4*j+2 : 4*j+4]

			# One-index notation for on-site interactions.
			if r_i == r_j:
				self.site[r_i] = self.hopp[r_j, r_j]
				self.pair[r_i] = self.pair[r_j, r_j]

	def asarray(self, copy=False):
		"""Represent the Hamiltonian of the system as a dense array.

		This method also ensures that the matrix is finalized, i.e. that e.g.
		particle-hole and hopping symmetries are ensured. One should always
		explicitly call `.asarray()` to access Hamiltonian matrix, as using
		the `.data` field directly may leave matrix elements out-of-date.
		"""
		for r_i, r_j in self.lattice.relevant():
			# Convert indices from coordinate form.
			i, j = self.lattice[r_i], self.lattice[r_j]

			# Diagonal particle-hole symmetry.
			self.data[4*i+2 : 4*i+4, 4*j+2 : 4*j+4] = \
				-self.data[4*i+0 : 4*i+2, 4*j+0 : 4*j+2].conj()

			# Anomalous particle-hole symmetry.
			self.data[4*i+2 : 4*i+4, 4*j+0 : 4*j+2] = \
				+self.data[4*i+0 : 4*i+2, 4*j+2 : 4*j+4].T.conj()
		
		for r_i, r_j in self.lattice.neighbors():
			# Convert indices from coordinate form.
			i, j = self.lattice[r_i], self.lattice[r_j]

			# Inverse neighbor hopping from symmetry.
			self.data[4*j+0 : 4*j+4, 4*i+0 : 4*i+4] = \
				+self.data[4*i+0 : 4*i+4, 4*j+0 : 4*j+4].T.conj()

		if not copy:
			return self.data
		else:
			return self.data.copy()
