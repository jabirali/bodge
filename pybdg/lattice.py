import numpy as np

from typing import Tuple

class Lattice:
	"""Representation of a cubic atomic lattice in three dimensions.

	This class provides convenience methods for iterating over all sites
	or neighbors in a lattice, as well as methods for converting between
	index (i1, i2) and coordinate ((x1, y1, z1), (x2, y2, z2)) notations.
	This simplfies the process of preparing lattice Hamiltonian matrices
	using flat indices while retaining the convenience of coordinates.
	"""

	def __init__(self, shape: Tuple[int, int, int]):
		self.shape = shape
		self.size = np.prod(shape)

	def __getitem__(self, index):
		"""Convert between coordinate and index notations."""
		return index[2] + index[1]*self.shape[2] + index[0]*self.shape[1]*self.shape[2]

	def sites(self):
		"""Generator for iterating over all sites in the lattice."""
		for x in range(self.shape[0]):
			for y in range(self.shape[1]):
				for z in range(self.shape[2]):
					yield (x, y, z)

	def neighbors(self, axis=None):
		"""Generator for iterating over all neighbors in the lattice.

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
			for x in range(self.shape[0]-1):
				for y in range(self.shape[1]):
					for z in range(self.shape[2]):
						yield (x, y, z), (x+1, y, z)
		elif axis == 1:
			# Neighbors along y-axis.
			for x in range(self.shape[0]):
				for y in range(self.shape[1]-1):
					for z in range(self.shape[2]):
						yield (x, y, z), (x, y+1, z)
		elif axis == 2:
			# Neighbors along z-axis.
			for x in range(self.shape[0]):
				for y in range(self.shape[1]):
					for z in range(self.shape[2]-1):
						yield (x, y, z), (x, y, z+1)

	def relevant(self):
		"""Generator for all relevant coordinate pairs in the lattice.

		This is useful when one might want to loop over both the on-site
		interactions (i, i) and nearest-neighbor interactions (i, j).
		"""
		for index in self.sites():
			yield (index, index)
		for indices in self.neighbors():
			yield indices