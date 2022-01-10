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
				H[i] = 1*σ3 + 2*σ2
				Δ[i] = 5*σ0 - 3*σ2

			for i, j in lattice.neighbors():
				H[i, j] = 3*σ0 - 4*σ2
				Δ[i, j] = 2*σ3 + 5*σ2
		
		# Verify that the result is Hermitian.
		assert np.allclose(system.data, system.data.T.conj())

	def test_eigenvectors(self):
		# Instantiate a system with superconductivity and a barrier.
		lattice = Cubic((10, 10, 1))
		system = System(lattice)

		with system as (H, Δ):
			for x, y, z in lattice.sites():
				H[x, y, z] = 4 * σ0
				if x > 5:
					Δ[x, y, z] = 1 * jσ2
				elif x > 3:
					H[x, y, z] += 6 * σ0
			for r1, r2 in lattice.neighbors():
				H[r1, r2] = -1 * σ0

		# Calculate the eigenvalues the manual way.
		E, X = eigh(system.data, subset_by_value=(0, np.inf))
		X = X.T

		# Confirm that we got positive eigenvalues and that we have
		# interpreted the corresponding eigenvector matrix correctly.
		assert E.size == 200
		for n, E_n in enumerate(E):
			assert E_n > 0
			assert np.allclose(system.data @ X[n, :], E_n * X[n, :])

		# Calculate the same eigenvalues via the package, and ensure
		# that the eigenvalues and eigenvectors are consistent.
		system.diagonalize()
		assert np.allclose(system.eigval, E)
		for n, E_n in enumerate(E):
			for m in range(100):
				assert np.allclose(system.eigvec[n, m, 0, 0], X[n, 4*m+0])
				assert np.allclose(system.eigvec[n, m, 0, 1], X[n, 4*m+1])
				assert np.allclose(system.eigvec[n, m, 1, 0], X[n, 4*m+2])
				assert np.allclose(system.eigvec[n, m, 1, 1], X[n, 4*m+3])
