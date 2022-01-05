import numpy as np

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
		lat = Lattice((3,5,7))
		sys = System(lat)

		for r in lat.sites():
			sys.hopp[r][:, :] = [[+1, -2j], [+2j, -1]]  # 1σ₃ + 2σ₂
			sys.pair[r][:, :] = [[+5, +3j], [-3j, +5]]  # 5σ₀ - 3σ₂

		for r1, r2 in lat.neighbors():
			sys.hopp[r1, r2][:, :] = [[+3, +4j], [-4j, +3]]  # 3σ₀ - 4σ₂
			sys.pair[r1, r2][:, :] = [[+2, -5j], [+5j, -2]]  # 2σ₃ + 5σ₂
		
		mat = sys.asarray()

		# Verify that the result is Hermitian.
		assert np.allclose(mat, mat.T.conj())
