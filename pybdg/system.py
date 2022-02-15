from matplotlib import style
from matplotlib.pyplot import colorbar
import numpy as np

from scipy.linalg import eigh

from scipy.sparse import coo_matrix
from scipy.sparse.bsr import bsr_matrix
from scipy.sparse.linalg import eigsh

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
	systems with particle-hole and spin degrees of freedom. Instead of
	explicitly constructing the matrix, this class allows you to specify the
	minimum number of matrix elements required via a `with` block, and the
	the rest are autofilled via symmetries. Moreover, it allows you to use
	`Lattice` coordinates instead of matrix indices to fill out the elements.
	"""
	def __init__(self, lattice: Cubic):
		# Lattice instance used as basis coordinates for the system.
		self.lattice = lattice

		# Number of lattice points that the system has. The integers in front of
		# the lattice size are the local degrees of freedom at each lattice site.
		self.shape = (4*lattice.size, 4*lattice.size)

		# Construct the most general 4N×4N Hamiltonian for the given lattice as
		# a sparse matrix. The fastest way to do so for a general lattice is
		# the COO format, so we use this to initialize the sparse matrix.
		pairs = (1+lattice.bonds)*lattice.size

		rows = np.zeros(pairs, dtype=np.int64)
		cols = np.zeros(pairs, dtype=np.int64)
		data = np.repeat(np.complex128(1), pairs)

		k = 0
		for _i, _j in lattice.relevant():
			i, j = 4*lattice[_i], 4*lattice[_j]

			rows[k] = i
			cols[k] = j
			k += 1

			if i != j:
				rows[k] = j
				cols[k] = i
				k += 1

		rows, cols, data = rows[:k], cols[:k], data[:k]
		self.matrix = coo_matrix((data, (rows, cols)))

		# Convert the matrix to the BSR format with 4x4 dense submatrices. This is the
		# most efficient format for handling matrix-vector multiplications in numerics.
		self.matrix = coo_matrix((data, (rows, cols)), shape=self.shape).tobsr((4, 4))

		# Discard the dummy entries used during matrix construction.
		self.matrix.data[...] = 0

	def __getitem__(self, keys):
		"""Accessor for 4x4 block at coordinates (row, col) of the Hamiltonian."""
		_i, _j = keys
		i, j = self.lattice[_i], self.lattice[_j]

		js = self.matrix.indices[self.matrix.indptr[i]:self.matrix.indptr[i+1]]
		k = self.matrix.indptr[i] + np.where(js == j)

		return self.matrix.data[k]

	def __setitem__(self, keys, val):
		"""Accessor for 4x4 block at coordinates (row, col) of the Hamiltonian."""
		_i, _j = keys
		i, j = self.lattice[_i], self.lattice[_j]

		js = self.matrix.indices[self.matrix.indptr[i]:self.matrix.indptr[i+1]]
		k = self.matrix.indptr[i] + np.where(js == j)

		self.matrix.data[k, ...] = val

	def __enter__(self):
		"""Implement a context manager interface for the class.

		This lets us write compact `with` blocks like the below, which is much
		more convenient than having to construct the matrix elements explicitly.

			>>> with system as (H, Δ):
			>>>     H[i, j] = ...
			>>>     Δ[i, j] = ...

		Note that the `__exit__` method is responsible for actually transferring
		all the elements of H and Δ to the correct locations in the Hamiltonian.
		"""
		self.hopp = {}
		self.pair = {}

		return self.hopp, self.pair

	def __exit__(self, exc_type, exc_val, exc_tb):
		"""Implement a context manager interface for the class.

		This part of the implementation takes care of finalizing the Hamiltonian:

		- Transferring elements from the context manager dicts to the actual matrix;
		- Ensuring that particle-hole and nearest-neighbor symmetries are respected;
		- Verifying that the constructed Hamiltonian is actually Hermitian.
		"""
		# Process hopping: H[i, j].
		for (i, j), val in self.hopp.items():
			Extract matrix block.
			self[i, j][0,0,0,0] = 100
			H = self[i, j]
			# print(H)

			# Set the electron-electron block.
			# self.data[4*i+0 : 4*i+2, 4*j+0 : 4*j+2] = +val

			# Set the hole-hole block.
			# self.data[4*i+2 : 4*i+4, 4*j+2 : 4*j+4] = -val.conj()

		# # Process pairing: Δ[i, j].
		# for (_i, _j), val in self.pair.items():
		# 	# Decode the coordinates.
		# 	i, j = self.lattice[_i], self.lattice[_j]

		# 	# Set the electron-hole block.
		# 	self.data[4*i+0 : 4*i+2, 4*j+2 : 4*j+4] = +val

		# 	# Set the hole-electron block.
		# 	self.data[4*i+2 : 4*i+4, 4*j+0 : 4*j+2] = +val.T.conj()

		# # Process inverse hopping.
		# for (_i, _j) in self.lattice.neighbors():
		# 	# Decode the coordinates.
		# 	i, j = self.lattice[_i], self.lattice[_j]

		# 	# Symmetry between hopping terms.
		# 	self.data[4*j+0 : 4*j+4, 4*i+0 : 4*i+4] = \
		# 		self.data[4*i+0 : 4*i+4, 4*j+0 : 4*j+4].T.conj()

		# # Verify that the matrix is Hermitian.
		# # if not np.allclose(self.data, self.data.T.conj()):
		# # 	raise RuntimeError("The constructed Hamiltonian is not Hermitian!")

		# # Scale the matrix so all eigenvalues are in (-1, +1).
		# # For numerical stability, we add a 1% safety margin.
		# self.scale = 1.01 * np.abs(eigsh(self.data, 1)[0][0])
		# self.data /= self.scale

		# # Reset accessors.
		# self.hopp = {}
		# self.pair = {}
		pass

	def index(self, row, col):
		"""Determine the sparse matrix index corresponding to block (row, col).

		This can be used to access `self.matrix.data[index, :, :]` when direct
		changes to the encapsulated BSR sparse matrix are required.
		"""
		indices, indptr = self.matrix.indices, self.matrix.indptr

		i, j = self.lattice[row], self.lattice[col]
		js = indices[indptr[i]:indptr[i+1]]
		k = indptr[i] + np.where(js == j)

		return k

	def diagonalize(self):
		"""Diagonalize the Hamiltonian of the system.

		This calculates the eigenvalues and eigenvectors of the system. Due to
		the particle-hole symmetry, only the positive eigenvalues are calculated.
		"""
		# Calculate the relevant eigenvalues and eigenvectors.
		self.eigval, self.eigvec = eigh(self.data, subset_by_value=(0, np.inf))

		# Restructure the eigenvectors to have the format eigvec[n, i, α],
		# where n corresponds to eigenvalue E[n], i is a position index, and
		# α represents the combined particle and spin index {e↑, e↓, h↑, h↓}.
		self.eigvec = self.eigvec.T.reshape((self.eigval.size, -1, 4))

	def plot(self):
		"""Visualize the sparsity structure of the generated matrix."""
		import matplotlib.pyplot as plt

		plt.figure(figsize=(8, 8))
		plt.spy(self.matrix, precision='present', markersize=1, marker='o', color='k')
		plt.title("Hamiltonian elements stored in the Block Sparse Row (BSR) representation")
		plt.xticks([4*i-0.5 for i in range(self.lattice.size)])
		plt.yticks([4*i-0.5 for i in range(self.lattice.size)])
		plt.grid()

		ax = plt.gca()
		ax.set_xticklabels([])
		ax.set_yticklabels([])

		plt.tight_layout()
		plt.show()