import numpy as np

from scipy.linalg import eigh
from scipy.sparse import coo_matrix, bsr_matrix
from scipy.sparse.linalg import norm

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

	Internally, this constructs a block-sparse matrix (BSR format), enabling
	the construction of very large physical systems (e.g. 10⁶ lattice points).
	"""
	def __init__(self, lattice: Cubic):
		# Lattice instance used as basis coordinates for the system.
		self.lattice = lattice

		# Number of lattice points that the system has. The integers in front of
		# the lattice size are the local degrees of freedom at each lattice site.
		self.shape = (4*lattice.size, 4*lattice.size)

		# Scale factor used to compress the Hamiltonian spectrum to (-1, +1).
		# This must be set to an upper bound for the Hamiltonian spectral radius.
		self.scale = 1.0

		# Initialize the most general 4N×4N Hamiltonian for this lattice as a
		# sparse matrix. The fastest alternative for this is the COO format.
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
		self.matrix = coo_matrix((data, (rows, cols)), shape=self.shape)

		# Convert the matrix to the BSR format with 4x4 dense submatrices. This is the
		# most efficient format for handling matrix-vector multiplications numerically.
		# We can then discard all the dummy entries used during matrix construction.
		self.matrix = self.matrix.tobsr((4, 4))
		self.matrix.data[...] = 0

		# Simplify direct access to the underlying data structure.
		self.data = self.matrix.data

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
		# Restore the Hamiltonian energy scale.
		self.data *= self.scale

		# Prepare storage for the context manager.
		self.hopp = {}
		self.pair = {}

		return self.hopp, self.pair

	def __exit__(self, exc_type, exc_val, exc_tb):
		"""Implement a context manager interface for the class.

		This part of the implementation takes care of finalizing the Hamiltonian:

		- Transferring elements from the context manager dicts to the sparse matrix;
		- Ensuring that particle-hole and nearest-neighbor symmetries are respected;
		- Verifying that the constructed Hamiltonian is actually Hermitian;
		- Scaling the Hamiltonian to have a spectrum bounded by (-1, +1).
		"""
		# Process hopping: H[i, j].
		for (i, j), val in self.hopp.items():
			# Find this matrix block.
			k = self.index(i, j)

			# Update electron-electron and hole-hole parts.
			self.data[k, 0:2, 0:2] = +val
			self.data[k, 2:4, 2:4] = -val.conj()

		# Process pairing: Δ[i, j].
		for (i, j), val in self.pair.items():
			# Find this matrix block.
			k = self.index(i, j)

			# Update electron-hole and hole-electron parts.
			self.data[k, 0:2, 2:4] = +val
			self.data[k, 2:4, 0:2] = +val.T.conj()

		# Process inverse hopping.
		for (i, j) in self.lattice.neighbors():
			# Find these matrix blocks.
			k1 = self.index(i, j)
			k2 = self.index(j, i)

			# Enforce symmetry of hopping terms.
			self.data[k2, ...] = np.swapaxes(self.data[k1, ...], 2, 3).conj()

		# Verify that the matrix is Hermitian.
		if np.max(self.matrix - self.matrix.getH()) > 1e-6:
			raise RuntimeError("The constructed Hamiltonian is not Hermitian!")

		# Scale the matrix so all eigenvalues are in (-1, +1). We here use
		# the theorem that the spectral radius is bounded by any matrix norm.
		self.scale = norm(self.matrix, 1)
		self.matrix /= self.scale

		# Reset accessors.
		self.hopp = {}
		self.pair = {}

	def index(self, row, col):
		"""Determine the sparse matrix index corresponding to block (row, col).

		This can be used to access `self.data[index, :, :]` when direct
		changes to the encapsulated block-sparse matrix are required.
		"""
		indices, indptr = self.matrix.indices, self.matrix.indptr

		i, j = self.lattice[row], self.lattice[col]
		js = indices[indptr[i]:indptr[i+1]]
		k = indptr[i] + np.where(js == j)

		return k

	def diagonalize(self):
		"""Diagonalize the Hamiltonian of the system.

		This calculates the eigenvalues and eigenvectors of the system. Due to
		the particle-hole symmetry, only positive eigenvalues are calculated.

		Note that this method is quite inefficient since it uses dense matrices;
		it is meant as a benchmark, not for actual large-scale calculations.
		"""
		# Calculate the relevant eigenvalues and eigenvectors.
		eigval, eigvec = eigh(self.matrix.todense(), subset_by_value=(0, np.inf))

		# Restructure the eigenvectors to have the format eigvec[n, i, α],
		# where n corresponds to eigenvalue E[n], i is a position index, and
		# α represents the combined particle and spin index {e↑, e↓, h↑, h↓}.
		eigvec = eigvec.T.reshape((eigval.size, -1, 4))

		return eigval, eigvec

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