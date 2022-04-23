from typing import Iterable

import numpy as np

from .consts import *


class Lattice:
    """Representation of a general three-dimensional lattice.

    This is an abstract class which defines an API for iterating over all the
    sites (atoms) and bonds (nearest neighbors) in a lattice. In the language
    of graph theory, this class lets us traverse all the nodes and links of a
    simple graph. The actual graph traversal must be defined by subclassing
    `Lattice` and implementing all methods that raise `NotImplementedError`.
    Note that the `.bonds` method should yield the links in a "sorted" way:
    i.e. it should yield ((0,0,0), (0,0,1)) but not ((0,0,1), (0,0,0)).

    Note that you are free to implement optional arguments to these methods.
    For instance, it may be useful to iterate over one sublattice at a time
    when calling `.sites` on a honeycomb lattice, or to iterate over the
    x- and y-axes separately when calling `.bonds` on a rectangular lattice.
    However, it must be possible to call both methods without additional
    arguments to traverse all sites and bonds in the lattice, respectively.
    """

    def __init__(self, shape: Coord):
        # Number of atoms per lattice dimension.
        self.shape: Coord = shape

        # Number of atoms in the lattice.
        self.size: Index = np.prod(shape)

        # Number of nearest neighbors per atom.
        self.ligancy: int = np.sum([2 for s in self.shape if s > 1], dtype=np.int64)

    def __getitem__(self, coord: Coord) -> Index:
        """Syntactic sugar for converting coordinates into indices."""
        return self.index(coord)

    def index(self, coord: Coord) -> Index:
        """Convert a 3D site coordinate to a 1D index."""
        raise NotImplementedError

    def sites(self) -> Iterable[Coord]:
        """Iterate over all atomic sites in the lattice."""
        raise NotImplementedError

    def bonds(self) -> Iterable[Coords]:
        """Iterate over all atomic bonds in the lattice."""
        raise NotImplementedError

    def terms(self) -> Iterable[Coords]:
        """Iterate over all interactions in the lattice."""
        for index in self.sites():
            yield (index, index)
        for indices in self.bonds():
            yield indices


class CubicLattice(Lattice):
    """Concrete representation of a cubic lattice."""

    def __init__(self, shape: Coord):
        # Initialize superclass.
        super().__init__(shape)

    def index(self, coord: Coord) -> Index:
        return coord[2] + coord[1] * self.shape[2] + coord[0] * self.shape[1] * self.shape[2]

    def sites(self) -> Iterable[Coord]:
        for x in range(self.shape[0]):
            for y in range(self.shape[1]):
                for z in range(self.shape[2]):
                    yield (x, y, z)

    def bonds(self, axis=None) -> Iterable[Coords]:
        if axis is None:
            # Neighbors along all axes.
            yield from self.bonds(axis=2)
            yield from self.bonds(axis=1)
            yield from self.bonds(axis=0)
        elif axis == 0:
            # Neighbors along x-axis.
            for x in range(self.shape[0] - 1):
                for y in range(self.shape[1]):
                    for z in range(self.shape[2]):
                        yield (x, y, z), (x + 1, y, z)
        elif axis == 1:
            # Neighbors along y-axis.
            for x in range(self.shape[0]):
                for y in range(self.shape[1] - 1):
                    for z in range(self.shape[2]):
                        yield (x, y, z), (x, y + 1, z)
        elif axis == 2:
            # Neighbors along z-axis.
            for x in range(self.shape[0]):
                for y in range(self.shape[1]):
                    for z in range(self.shape[2] - 1):
                        yield (x, y, z), (x, y, z + 1)
