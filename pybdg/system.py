import numpy as np

from scipy.linalg import eigh

from .lattice import Lattice


# Define the Pauli matrices used to represent spin.
σ0 = np.array([[+1,  0 ], [ 0, +1]], dtype=np.complex64)
σ1 = np.array([[ 0, +1 ], [+1,  0]], dtype=np.complex64)
σ2 = np.array([[ 0, -1j], [+1j, 0]], dtype=np.complex64)
σ3 = np.array([[+1,  0 ], [ 0, -1]], dtype=np.complex64)

jσ0 = 1j * σ0
jσ1 = 1j * σ1
jσ2 = 1j * σ2
jσ3 = 1j * σ3


class System:
	"""Representation of a physical system in the tight-binding limit.

	This can be used to construct Hamiltonian matrices for condensed matter
	systems that exhibit particle-hole and spin degrees of freedom. Instead
	of explicitly constructing the whole matrix, this class allows you to
	specify the minimum number of matrix elements required as the rest are
	autofilled via symmetries. Moreover, it allows you to use `Lattice`
	coordinates instead of matrix indices to fill out these elements.
	"""
	def __init__(self, lattice: Lattice):
		# Lattice instance used as basis coordinates for the system.
		self.lattice = lattice

		# Number of lattice points that the system has. The integers in front of
		# the lattice size are the local degrees of freedom at each lattice site.
		self.shape = (4*lattice.size, 4*lattice.size)

		# Hamiltonian matrix used in the tight-binding treatment.
		self.data = np.zeros(self.shape, dtype=np.complex128)

		# Convenience accessors for electron-electron and electron-hole blocks
		# in the Hamiltonian matrix at each site. Note that the hole-electron
		# and hole-hole blocks are autofilled using the Hamiltonian symmetries.
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
				self.hopp[r_i] = self.hopp[r_j, r_j]
				self.pair[r_i] = self.pair[r_j, r_j]

		# TODO: Create separate places to store the physical fields.

	def finalize(self):
		"""Represent the Hamiltonian of the system as a dense array.

		This method also ensures that the matrix is finalized, i.e. that e.g.
		particle-hole and hopping symmetries are ensured. One should always
		explicitly call this method to access Hamiltonian matrix, as using
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

		# Scaling due to adding particle-hole symmetry.
		self.data /= 2

		# Verify that the matrix is Hermitian.
		if not np.allclose(self.data, self.data.T.conj()):
			raise RuntimeError("Error: Hamiltonian is not Hermitian!")

		return self.data

	def diagonalize(self):
		"""Diagonalize the Hamiltonian of the system.

		This calculates the eigenvalues and eigenvectors of the system. Due to
		the particle-hole symmetry, only the positive eigenvalues are calculated.
		"""
		# Calculate the relevant eigenvalues and eigenvectors.
		self.eigval, self.eigvec = eigh(self.data, subset_by_value=(0, np.inf))

		# Restructure the eigenvectors to have the format eigvec[n, i, e, s],
		# where n corresponds to eigenvalue E[n], i is a position index, e is
		# electron (0) or hole (1), and s is spin-up (0) or spin-down (1).
		self.eigvec = self.eigvec.T.reshape((self.eigval.size, -1, 2, 2))

	def green(self):
		"""Calculate the single-particle Green function."""
		pass
