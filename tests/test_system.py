import numpy as np

from pybdg.lattice import *
from pybdg.system import *

class TestSystem:
	def test_pauli(self):
		# Test that the quaternion identities hold.
		assert np.allclose(σˣ @ σˣ, σᵒ)
		assert np.allclose(σʸ @ σʸ, σᵒ)
		assert np.allclose(σᶻ @ σᶻ, σᵒ)

		assert np.allclose(σˣ @ σʸ, 1j * σᶻ)
		assert np.allclose(σʸ @ σᶻ, 1j * σˣ)
		assert np.allclose(σᶻ @ σˣ, 1j * σʸ)

		assert np.allclose(σˣ @ σʸ @ σᶻ, 1j * σᵒ)

	def test_hermitian(self):
		# Instantiate a relatively dense complex Hamiltonian.
		lat = Lattice((3,5,7))
		sys = System(lat)

		for r in lat.sites():
			sys.elec[r][:, :] = [[+1, -2j], [+2j, -1]]  # 1σ₃ + 2σ₂
			sys.pair[r][:, :] = [[+5, +3j], [-3j, +5]]  # 5σ₀ - 3σ₂

		for r1, r2 in lat.neighbors():
			sys.elec[r1, r2][:, :] = [[+3, +4j], [-4j, +3]]  # 3σ₀ - 4σ₂
			sys.pair[r1, r2][:, :] = [[+2, -5j], [+5j, -2]]  # 2σ₃ + 5σ₂
		
		mat = sys.asarray()

		# Verify that the result is Hermitian.
		assert np.allclose(mat, mat.T.conj())
