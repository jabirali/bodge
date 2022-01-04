class Lattice:
	"""Representation of a cubic lattice.

	This class provides convenience methods for iterating over all sites
	or neighbors in a lattice, as well as methods for converting between
	index (i1, i2) and coordinate ((x1, y1, z1), (x2, y2, z2)) notations.
	This simplfies the process of preparing lattice Hamiltonian matrices
	using flat indices while retaining the convenience of coordinates.
	"""

	def __init__(self, dims):
		self.dims = dims

	def __getitem__(self, inds):
		"""Convert between coordinate and index notations."""
		return inds[2] + inds[1]*self.dims[2] + inds[0]*self.dims[1]*self.dims[2]

	def sites(self):
		"""Generator for site coordinates in the lattice."""
		for x in range(self.dims[0]):
			for y in range(self.dims[1]):
				for z in range(self.dims[2]):
					yield (x, y, z)

	def neighbors(self, axis=None):
		"""Generator for nearest-neighbor coordinates in the lattice.

		The argument `axis` specifies whether we are only interested in
		nearest-neighbor pairs along a certain direction, which is useful
		when e.g. specifying hopping terms for spin-orbit coupling.

		Note that this returns pairs only in the "increasing" direction.
		For instance, it yields ((0,0,0), (0,0,1)) but ((0,0,1), (0,0,0)),
		such that any inverse hopping terms must be dealt with yourself.
		"""
		if axis is None:
			# Neighbors along all axes.
			yield from self.neighbors(axis=2)
			yield from self.neighbors(axis=1)
			yield from self.neighbors(axis=0)
		elif axis == 0:
			# Neighbors along x-axis.
			for x in range(self.dims[0]-1):
				for y in range(self.dims[1]):
					for z in range(self.dims[2]):
						yield (x, y, z), (x+1, y, z)
		elif axis == 1:
			# Neighbors along y-axis.
			for x in range(self.dims[0]):
				for y in range(self.dims[1]-1):
					for z in range(self.dims[2]):
						yield (x, y, z), (x, y+1, z)
		elif axis == 2:
			# Neighbors along z-axis.
			for x in range(self.dims[0]):
				for y in range(self.dims[1]):
					for z in range(self.dims[2]-1):
						yield (x, y, z), (x, y, z+1)

	def relevant(self):
		"""Generator for all relevant coordinate pairs in the lattice.

		This simplifies cases where one might want to loop over both on-site
		interactions (i, i) and nearest-neighbor interactions (i, j).
		"""
		for ind in self.sites():
			yield (ind, ind)
		for inds in self.neighbors():
			yield inds