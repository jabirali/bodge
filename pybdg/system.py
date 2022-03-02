import numpy as np

from multiprocessing import Pool
from scipy.linalg import eigh, inv
from scipy.sparse import coo_matrix, bsr_matrix, dia_matrix, identity, hstack
from scipy.sparse.linalg import norm
from tqdm import tqdm, trange
from rich import print

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
		print("[green]:: Preparing a sparse skeleton for the Hamiltonian[/green]")
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
		self.hamiltonian = coo_matrix((data, (rows, cols)), shape=self.shape)

		# Convert the matrix to the BSR format with 4x4 dense submatrices. This is the
		# most efficient format for handling matrix-vector multiplications numerically.
		# We can then discard all the dummy entries used during matrix construction.
		self.hamiltonian = self.hamiltonian.tobsr((4, 4))
		self.hamiltonian.data[...] = 0

		# Simplify direct access to the underlying data structure.
		self.data = self.hamiltonian.data

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

		print("[green]:: Collecting new contributions to the Hamiltonian[/green]")
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
		print("[green]:: Updating the matrix elements of the Hamiltonian[/green]")
		for (i, j), val in tqdm(self.hopp.items(), desc=' -> hopping', unit='', unit_scale=True):
			# Find this matrix block.
			k1 = self.index(i, j)

			# Update electron-electron and hole-hole parts.
			self.data[k1, 0:2, 0:2] = +val
			self.data[k1, 2:4, 2:4] = -val.conj()

			# Inverse process for non-diagonal contributions.
			if i != j:
				k2 = self.index(j, i)
				self.data[k2, ...] = np.swapaxes(self.data[k1, ...], 2, 3).conj()

		# Process pairing: Δ[i, j].
		for (i, j), val in tqdm(self.pair.items(), desc=' -> pairing', unit='', unit_scale=True):
			# Find this matrix block.
			k1 = self.index(i, j)

			# Update electron-hole and hole-electron parts.
			self.data[k1, 0:2, 2:4] = +val
			self.data[k1, 2:4, 0:2] = +val.T.conj()

			# Inverse process for non-diagonal contributions.
			if i != j:
				k2 = self.index(j, i)
				self.data[k2, ...] = np.swapaxes(self.data[k1, ...], 2, 3).conj()

		# Verify that the matrix is Hermitian.
		print(' -> checking that the matrix is hermitian')
		if np.max(self.hamiltonian - self.hamiltonian.getH()) > 1e-6:
			raise RuntimeError("The constructed Hamiltonian is not Hermitian!")

		# Scale the matrix so all eigenvalues are in (-1, +1). We here use
		# the theorem that the spectral radius is bounded by any matrix norm.
		print(' -> normalizing the spectral radius')
		self.scale = norm(self.hamiltonian, 1)
		self.hamiltonian /= self.scale

		# Reset accessors.
		print(' -> done!\n')
		self.hopp = {}
		self.pair = {}

	def index(self, row, col):
		"""Determine the sparse matrix index corresponding to block (row, col).

		This can be used to access `self.data[index, :, :]` when direct
		changes to the encapsulated block-sparse matrix are required.
		"""
		indices, indptr = self.hamiltonian.indices, self.hamiltonian.indptr

		i, j = self.lattice[row], self.lattice[col]
		js = indices[indptr[i]:indptr[i+1]]
		k = indptr[i] + np.where(js == j)

		return k

	@property
	def identity(self):
		"""Generate an identity matrix with similar dimensions as the Hamiltonian."""
		return identity(self.shape[1], 'int8').tobsr((4, 4))

	def diagonalize(self):
		"""Calculate the exact eigenstates of the system via direct diagonalization.

		This calculates the eigenvalues and eigenvectors of the system. Due to
		the particle-hole symmetry, only positive eigenvalues are calculated.

		Note that this method is quite inefficient since it uses dense matrices;
		it is meant as a benchmark, not for actual large-scale calculations.
		"""
		# Calculate the relevant eigenvalues and eigenvectors.
		print("[green]:: Calculating eigenstates via direct diagonalization[/green]")
		H = self.scale * self.hamiltonian.todense()
		eigval, eigvec = eigh(H, subset_by_value=(0, np.inf))

		# Restructure the eigenvectors to have the format eigvec[n, i, α],
		# where n corresponds to eigenvalue E[n], i is a position index, and
		# α represents the combined particle and spin index {e↑, e↓, h↑, h↓}.
		eigvec = eigvec.T.reshape((eigval.size, -1, 4))

		return eigval, eigvec

	def spectralize(self, energies, resolution=1e-4):
		"""Calculate the exact spectral function of the system via direct inversion."""
		# Restore the Hamiltonian scale and switch to dense matrices.
		H = self.scale * self.hamiltonian.todense()
		I = self.identity.todense()

		# The resolution is controlled by the imaginary energy.
		η = resolution * 1j

		# Calculate the spectral function via direct inversion.
		spectral = []
		print("[green]:: Calculating spectral function via direct inversion[/green]")
		for ω in tqdm(energies, desc=' -> energies', unit='', unit_scale=True):
			Gᴿ = inv((ω+η)*I - H)
			Gᴬ = inv((ω-η)*I - H)
			A = (1j/(2*np.pi)) * (Gᴿ - Gᴬ)

			spectral.append(A)

		return spectral

	def plot(self, grid=False):
		"""Visualize the sparsity structure of the generated matrix."""
		import matplotlib.pyplot as plt

		plt.figure(figsize=(8, 8))
		plt.spy(self.hamiltonian, precision='present', markersize=1, marker='o', color='k')
		plt.title("Hamiltonian elements stored in the Block Sparse Row (BSR) representation")
		plt.xticks([])
		plt.yticks([])

		if grid:
			plt.xticks([4*i-0.5 for i in range(self.lattice.size)])
			plt.yticks([4*i-0.5 for i in range(self.lattice.size)])
			plt.grid()

		ax = plt.gca()
		ax.set_xticklabels([])
		ax.set_yticklabels([])

		plt.tight_layout()
		plt.show()

class Chebyshev:
	"""This class facilitates a Chebyshev expansion of the Green functions.

	The `radius` determines the size of the Local Krylov subspace used for the
	expansion, `moments` sets the number of Chebyshev matrices to include in
	the expansion, and `system` provides a previously configured Hamiltonian.
	"""
	def __init__(self, system, moments=100, radius=4):
		# Chebyshev nodes {ω_m} where we will calculate the Green function.
		k = np.arange(2*moments)
		ω = np.cos(np.pi * (2*k + 1) / (4*moments))

		# Calculate the corresponding Chebyshev transform coefficients.
		# TODO: Incorporate the relevant Lorentz kernel factors here.
		n = np.arange(moments)
		T = (2.0/np.pi) * np.cos(n[None,:] * np.arccos(ω[:, None])) / np.sqrt(1 - ω[:, None]**2)
		T[:, 0] /= 2

		# Determine block size and number of blocks from the subspace radius.
		if radius < 1:
			raise RuntimeError("Invalid radius. The subspace radius should be at least one.")

		blocksize = 2 ** round(np.log2(self.shape[1] / radius**3))
		blocks = self.shape[1] // blocksize

		if blocks * blocksize != self.shape[1]:
			raise RuntimeError("Invalid blocksize. Is the system dimension a power of two?")

		# Prepare a cheap surrogate from the Hamiltonian.
		H = bsr_matrix(self.hamiltonian, dtype=np.int8)
		H.data[...] = 1

		# Prepare matrices for the subspace expansion.
		diag = np.repeat(np.int8(1), blocksize)


		# Save relevant variables internally.
		self.hamiltonian = system.hamiltonian
		self.moments = moments
		self.radius = radius

		self.transform = T
		self.energies = ω

	def __call__(self, args):
		"""Chebyshev expansion of a given Green function block."""
		# Unpack the provided arguments
		I_k, H_k, P_k = args

		# Shorter names for stored stuff.
		H = self.H
		N = self.N
		T = self.T

		# Initialize the first two Chebyshev matrices needed to start recursion.
		G_k0 = I_k
		G_kn = H @ I_k

		# Green function slices G_k(ω_m) at the Chebyshev nodes ω_m. These are
		# initialized using the first two Chebyshev matrices defined above. No
		# projection is needed here since H_k and G_kn have the same structure.
		G_k = [T[m, 0] * G_k0 + T[m, 1] * G_kn for m in range(2*N)]

		# Multiply the projection operator by 2x to fit the recursion relation.
		P_k *= 2

		# Chebyshev expansion of the next elements.
		for n in range(2, N):
			# Chebyshev expansion of next vector. Element-wise multiplication
			# by P_k projects the result back into the Local Krylov subspace.
			G_kn, G_k0 = (H @ G_kn).multiply(P_k) - G_k0, G_kn

			# Perform the Chebyshev transformation. Element-wise multiplication
			# by H_k preserves only on-site and nearest-neighbor interactions.
			# WARNING: This has been optimized to ignore SciPy wrapper checks.
			GH_kn = G_kn.multiply(H_k)
			for m, G_km in enumerate(G_k):
				G_km.data += T[m, n] * GH_kn.data

		return G_k

	def subspace(self):
		"""Generator used for Local Krylov subspace expansion of matrix polynomials.

		For a given radius R, this function determines an appropriate block size M,
		such that 4N×4N matrices are split into 4N×4M matrices with M≤N. For each
		such block, the generator then returns the following matrices:
		* I_k [4N×4M] is the k'th slice of the full identity matrix [4N×4N]. This
		  can be used as a starting point for e.g. iterative inversion of matrices.
		* H_k [4N×4M] has ones where the k'th slice of the Hamiltonian has non-zero
		  elements and zeros elsewhere. Element-wise multiplication by this matrix
		  will discard all terms but on-site and nearest-neighbor interactions.
		* P_k [4N×4M] has one-blocks where the k'th slice of H^R has nonzero elements
		  and zeros elsewhere. Element-wise multiplication by this matrix projects
		  matrices onto the Local Krylov subspace, effectively discarding all matrix
		  elements corresponding to (R+1)'th nearest-neighbor interactions and up.

		The blocksize M is determined such that the density of a general matrix G in
		the Local Krylov subspace P_m * G is similar to the sparse Hamiltonian H.
		Since the volume of the subspace is V = (4/3)πR^3 ≈ 4R³, we can set the
		blocksize to M ≈ N/V = N/4R³, resulting in a matrix width of 4M = N/R³.
		"""
		# Prepare a cheap surrogate from the Hamiltonian.
		H = bsr_matrix(self.hamiltonian, dtype=np.int8)
		H.data[...] = 1

		# Prepare matrices for the subspace expansion.
		diag = np.repeat(np.int8(1), blocksize)
		for k in trange(blocks, unit='block'):
			# Generate the k'th slice of the identity matrix.
			I_k = dia_matrix((diag, [-k*blocksize]), (self.shape[0], blocksize)).tobsr((4, 4))

			# Generate the k'th slice of the Hamiltonian mask.
			H_k = H @ I_k
			H_k.data[...] = 1

			# Generate the k'th slice of the projection mask.
			P_k = H_k
			for r in range(2, radius):
				P_k = H @ P_k
			P_k.data[...] = 1

			yield (I_k, H_k, P_k)