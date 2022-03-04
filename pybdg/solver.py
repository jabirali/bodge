from scipy.sparse import bsr_matrix, dia_matrix
from joblib import Parallel, delayed

import numpy as np

class Solver:
	"""This class facilitates a Chebyshev expansion of the Green functions.

	The `radius` determines the size of the Local Krylov subspace used for the
	expansion, `moments` sets the number of Chebyshev matrices to include in
	the expansion, and `system` provides a previously configured Hamiltonian.
	"""
	def __init__(self, system, moments=200, radius=4, blocksize=512):
		# Sanity checks for the arguments.
		if radius < 1:
			raise RuntimeError("Invalid radius: Must be a positive integer.")
		if system.shape[1] % blocksize != 0:
			raise RuntimeError("Invalid blocksize: Must divide the Hamiltonian dimension.")

		# Chebyshev nodes {ω_m} where we will calculate the Green function.
		k = np.arange(2*moments)
		ω = np.cos(np.pi * (2*k + 1) / (4*moments))

		# Calculate the corresponding Chebyshev transform coefficients.
		# TODO: Incorporate the relevant Lorentz kernel factors here.
		n = np.arange(moments)
		T = (2.0/np.pi) * np.cos(n[None,:] * np.arccos(ω[:, None])) / np.sqrt(1 - ω[:, None]**2)
		T[:, 0] /= 2

		# Prepare a cheap surrogate from the Hamiltonian.
		H = bsr_matrix(system.hamiltonian, dtype=np.int8)
		H.data[...] = 1
		self.H0 = H

		# Prepare matrices for the subspace expansion.
		self.diag = np.repeat(np.int8(1), blocksize)


		# Save relevant variables internally.
		self.hamiltonian = system.hamiltonian
		self.moments = moments
		self.radius = radius

		self.transform = T
		self.energies = ω

		self.blocksize = blocksize
		self.blocks = system.shape[1] // blocksize

	def __call__(self, block):
		"""Chebyshev expansion of a given Green function block."""
		# Compact notation for relevant variables.
		k = block
		M = self.H0

		# Generate a slice of the identity matrix.
		I_k = dia_matrix((self.diag, [-block*self.blocksize]), (self.hamiltonian.shape[0], self.blocksize)).tobsr((4, 4))

		# Generate the k'th slice of the Hamiltonian mask.
		H_k = M @ I_k
		H_k.data[...] = 1

		# Generate the k'th slice of the projection mask.
		P_k = H_k
		for r in range(2, self.radius):
			P_k = M @ P_k
		P_k.data[...] = 1

		# Shorter names for stored stuff.
		H = self.hamiltonian
		N = self.moments
		T = self.transform

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

	def run(self, proc=4):
		G = Parallel(n_jobs=proc, backend='loky')(delayed(self)(block) for block in range(self.blocks))

		# Transpose G

		# Hstack G

		return G