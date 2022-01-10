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
		lat = Cubic((3,5,7))
		sys = System(lat)

		for r in lat.sites():
			sys.hopp[r][:, :] = [[+1, -2j], [+2j, -1]]  # 1σ₃ + 2σ₂
			sys.pair[r][:, :] = [[+5, +3j], [-3j, +5]]  # 5σ₀ - 3σ₂

		for r1, r2 in lat.neighbors():
			sys.hopp[r1, r2][:, :] = [[+3, +4j], [-4j, +3]]  # 3σ₀ - 4σ₂
			sys.pair[r1, r2][:, :] = [[+2, -5j], [+5j, -2]]  # 2σ₃ + 5σ₂
		
		mat = sys.finalize()

		# Verify that the result is Hermitian.
		assert np.allclose(mat, mat.T.conj())

	def test_eigenvectors(self):
		# Instantiate a system with superconductivity and a barrier.
		lat = Cubic((10, 10, 1))
		sys = System(lat)

		for x, y, z in lat.sites():
			sys.hopp[x, y, z] += 4 * σ0
			if x > 5:
				sys.pair[x, y, z] += 1 * jσ2
			elif x > 3:
				sys.hopp[x, y, z] += 6 * σ0
		for r1, r2 in lat.neighbors():
			sys.hopp[r1, r2] = -1 * σ0

		# Create two copies of the Hamiltonian.
		H = sys.finalize()

		# Calculate the eigenvalues the manual way.
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
		sys.diagonalize()
		assert np.allclose(sys.eigval, E)
		for n, E_n in enumerate(E):
			for m in range(100):
				assert np.allclose(sys.eigvec[n, m, 0, 0], X[n, 4*m+0])
				assert np.allclose(sys.eigvec[n, m, 0, 1], X[n, 4*m+1])
				assert np.allclose(sys.eigvec[n, m, 1, 0], X[n, 4*m+2])
				assert np.allclose(sys.eigvec[n, m, 1, 1], X[n, 4*m+3])
