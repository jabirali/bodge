import numpy as np

from scipy.linalg import eigh

from .lattice import Cubic


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
	def __init__(self, lattice: Cubic):
		# Lattice instance used as basis coordinates for the system.
		self.lattice = lattice

		# Number of lattice points that the system has. The integers in front of
		# the lattice size are the local degrees of freedom at each lattice site.
		self.shape = (4*lattice.size, 4*lattice.size)

		# Hamiltonian matrix used in the tight-binding treatment.
		self.data = np.zeros(self.shape, dtype=np.complex128)

		# Placeholders for accesors.
		self.hopp = {}
		self.pair = {}

	def __enter__(self):
		"""Implement a context manager interface for the class.

		This lets us write compact `with` blocks like the below, which is more
		convenient than explicitly referring to e.g. `system.pair[i, j][...]`.

			>>> with system as (H, Δ):
			>>>     H[i, j][...] = ...
			>>>     Δ[i, j][...] = ...
		"""
		return self.hopp, self.pair

	def __exit__(self, exc_type, exc_val, exc_tb):
		"""Implement a context manager interface for the class.

		This part of the implementation takes care of finalizing the Hamiltonian
		matrix, which means that the symmetries of the matrix are taken care of.
		"""
		# Process hopping: H[i, j].
		for key, val in self.hopp.items():
			# Decode the coordinates.
			i, j = self.lattice[key[0]], self.lattice[key[1]]

			# Set the electron-electron block.
			self.data[4*i+0 : 4*i+2, 4*j+0 : 4*j+2] = +val / 2

			# Set the hole-hole block.
			self.data[4*i+2 : 4*i+4, 4*j+2 : 4*j+4] = -val.conj() / 2

		# Process pairing: Δ[i, j].
		for key, val in self.pair.items():
			# Decode the coordinates.
			i, j = self.lattice[key[0]], self.lattice[key[1]]

			# Set the electron-hole block.
			self.data[4*i+0 : 4*i+2, 4*j+2 : 4*j+4] = +val / 2

			# Set the hole-electron block.
			self.data[4*i+2 : 4*i+4, 4*j+0 : 4*j+2] = +val.T.conj() / 2

		# Process inverse hopping.
		for key in self.lattice.neighbors():
			# Decode the coordinates.
			i, j = self.lattice[key[0]], self.lattice[key[1]]

			# Symmetry between hopping terms.
			self.data[4*j+0 : 4*j+4, 4*i+0 : 4*i+4] = \
				+self.data[4*i+0 : 4*i+4, 4*j+0 : 4*j+4].T.conj()

		# Verify that the matrix is Hermitian.
		if not np.allclose(self.data, self.data.T.conj()):
			raise RuntimeError("Error: Hamiltonian is not Hermitian!")

		# Reset accessors.
		self.hopp = {}
		self.pair = {}

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
