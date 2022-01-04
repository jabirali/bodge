import numpy as np

from pybdg.lattice import *
from pybdg.system import *

class TestSystem:
	def test_hermitian(self):
		# Instantiate a relatively dense complex Hamiltonian.
		lat = Lattice((3,5,7))
		sys = System(lat)

		for r in lat.sites():
			sys.ham[r][:, :] = [[+1, -2j], [+2j, -1]]  # 1σ₃ + 2σ₂
			sys.gap[r][:, :] = [[+5, +3j], [-3j, +5]]  # 5σ₀ - 3σ₂

		for r1, r2 in lat.neighbors():
			sys.ham[r1, r2][:, :] = [[+3, +4j], [-4j, +3]]  # 3σ₀ - 4σ₂
			sys.gap[r1, r2][:, :] = [[+2, -5j], [+5j, -2]]  # 2σ₃ + 5σ₂
		
		mat = sys.asarray()

		# Verify that the result is Hermitian.
		assert np.allclose(mat, mat.T.conj())
