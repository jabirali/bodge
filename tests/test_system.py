import numpy as np

from scipy.linalg import eigh

from pybdg.lattice import *
from pybdg.system import *

class TestSystem:
	def test_pauli(self):
		# Test that the quaternion identities hold.
		assert np.allclose(σ1 @ σ1, σ0)
		assert np.allclose(σ2 @ σ2, σ0)
		assert np.allclose(σ3 @ σ3, σ0)

		assert np.allclose(σ1 @ σ2, jσ3)
		assert np.allclose(σ2 @ σ3, jσ1)
		assert np.allclose(σ3 @ σ1, jσ2)

		assert np.allclose(σ1 @ σ2 @ σ3, jσ0)

	def test_hermitian(self):
		# Instantiate a somewhat dense complex Hamiltonian. Note that
		# the submatrices need to be Hermitian for the whole to be so.
		lattice = Cubic((3,5,7))
		system = System(lattice)

		with system as (H, Δ):
			for i in lattice.sites():
				H[i, i] = 1*σ3 + 2*σ2
				Δ[i, i] = 5*σ0 - 3*σ2

			for i, j in lattice.neighbors():
				H[i, j] = 3*σ0 - 4*σ2
				Δ[i, j] = 2*σ3 + 5*σ2
		
		# Verify that the result is Hermitian.
		H = system.hamiltonian.todense()
		assert np.allclose(H, H.T.conj())

	def test_eigenvectors(self):
		# Instantiate a system with superconductivity and a barrier.
		lattice = Cubic((10, 10, 1))
		system = System(lattice)

		with system as (H, Δ):
			for i in lattice.sites():
				H[i, i] = 4 * σ0
				if i[0] > 5:
					Δ[i, i] = 1 * jσ2
				elif i[0] > 3:
					H[i, i] += 6 * σ0
			for i, j in lattice.neighbors():
				H[i, j] = -1 * σ0

		# Calculate the eigenvalues the manual way.
		H = system.scale * system.hamiltonian.todense()
		E, X = eigh(H, subset_by_value=(0, np.inf))
		X = X.T

		# Confirm that we got positive eigenvalues and that we have
		# interpreted the corresponding eigenvector matrix correctly.
		assert E.size == 200
		for n, E_n in enumerate(E):
			assert E_n > 0
			assert np.allclose(H @ X[n, :], E_n * X[n, :])

		# Calculate the same eigenvalues via the package, and ensure
		# that the eigenvalues and eigenvectors are consistent.
		eigval, eigvec = system.diagonalize()
		assert np.allclose(eigval, E)
		for n, E_n in enumerate(E):
			for m in range(100):
				assert np.allclose(eigvec[n, m, 0], X[n, 4*m+0])
				assert np.allclose(eigvec[n, m, 1], X[n, 4*m+1])
				assert np.allclose(eigvec[n, m, 2], X[n, 4*m+2])
				assert np.allclose(eigvec[n, m, 3], X[n, 4*m+3])

	def test_sparsity(self):
		# Instantiate a somewhat random test system.
		lattice = Cubic((3,5,7))
		system = System(lattice)

		with system as (H, Δ):
			for i in lattice.sites():
				H[i, i] = 1*σ3 + 2*σ2
				Δ[i, i] = 5*σ0 - 3*σ2

			for i, j in lattice.neighbors():
				H[i, j] = 3*σ0 - 4*σ2
				Δ[i, j] = 2*σ3 + 5*σ2

		# Calculate a matrix product using internal matrices.
		H = system.hamiltonian
		I = system.identity
		G = H @ I

		# Ensure that the Hamiltonian H has a 4x4 BSR representation,
		# and that the identity I preserves both the value and format.
		assert H.blocksize == (4, 4)
		assert G.blocksize == (4, 4)
		assert np.allclose(G.todense(), H.todense())
