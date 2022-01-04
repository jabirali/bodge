import numpy as np

class System:
	"""Representation of a physical system in the tight-binding limit.

	This class is used to construct Hamiltonian matrices for condensed matter
	systems that have particle-hole and spin degrees of freedom in the tight-
	binding approximation. It provides a convenient interface to specifying
	both normal (electronic) and anomalous (superconducting) contributions,
	and then exploits symmetries to autofill the remaining matrix elements.
	"""
	def __init__(self, lat):
		# Lattice instance used as coordinates for the system.
		self.lat = lat

		# Number of lattice points that the system has in total.
		self.dim = np.prod(lat.dims)

		# Hamiltonian datarix used in the tight-binding treatment. This is
		# a 4N×4N datarix, where we assume particle-hole and spin degrees
		# of freedom, and the remaining N is the number of lattice points.
		self.data = np.zeros((4*self.dim, 4*self.dim), dtype=np.complex64)

		# Simplified accessors for the diagonal and anomalous datarix parts.
		# Only these need to be explicitly defined, since the class is smart
		# enough to construct the remaining terms via symmetry considerations.
		self.ham = {}
		self.gap = {}
		for r_i, r_j in self.lat.relevant():
			# Convert indices from coordinate form.
			i, j = self.lat[r_i], self.lat[r_j]

			# Two-index notation for every interaction.
			self.ham[r_i, r_j] = self.data[4*i+0 : 4*i+2, 4*j+0 : 4*j+2]
			self.gap[r_i, r_j] = self.data[4*i+0 : 4*i+2, 4*j+2 : 4*j+4]

			# One-index notation for on-site interactions.
			if r_i == r_j:
				self.ham[r_i] = self.ham[r_j, r_j]
				self.gap[r_i] = self.gap[r_j, r_j]

	def asarray(self, copy=False):
		"""Represent the Hamiltonian of the system as a dense array.

		This method also ensures that the matrix is finalized, i.e. that e.g.
		particle-hole and hopping symmetries are ensured. One should always
		explicitly call `.asarray()` to access Hamiltonian matrix, as using
		the `.data` field directly may leave matrix elements out-of-date.
		"""
		for r_i, r_j in self.lat.relevant():
			# Convert indices from coordinate form.
			i, j = self.lat[r_i], self.lat[r_j]

			# Diagonal particle-hole symmetry.
			self.data[4*i+2 : 4*i+4, 4*j+2 : 4*j+4] = \
				-self.data[4*i+0 : 4*i+2, 4*j+0 : 4*j+2].conj()

			# Anomalous particle-hole symmetry.
			self.data[4*i+2 : 4*i+4, 4*j+0 : 4*j+2] = \
				+self.data[4*i+0 : 4*i+2, 4*j+2 : 4*j+4].T.conj()
		
		for r_i, r_j in self.lat.neighbors():
			# Convert indices from coordinate form.
			i, j = self.lat[r_i], self.lat[r_j]

			# Neighbor hopping symmetry.   
			self.data[4*j+0 : 4*j+4, 4*i+0 : 4*i+4] = \
				+self.data[4*i+0 : 4*i+4, 4*j+0 : 4*j+4].T.conj()

		if not copy:
			return self.data
		else:
			return self.data.copy()