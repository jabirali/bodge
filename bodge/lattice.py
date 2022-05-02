from typing import Iterator, Optional

import numpy as np

from .consts import *

# Data types for working with Lattice coordinates.
Index = int
Coord = tuple[int, int, int]
Indices = tuple[Index, Index]
Coords = tuple[Coord, Coord]


class Lattice:
    """API for working with generic three-dimensional lattices.

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
        # Create an abstract class.
        if self.__class__.__name__ == "Lattice":
            raise ValueError("This class is not intended to be instantiated directly.")

        # Number of atoms per dimension.
        self.shape: Coord = shape

        # Number of atoms in the lattice.
        self.size: Index = np.prod(shape)

    def __getitem__(self, coord: Coord) -> Index:
        """Syntactic sugar for converting coordinates into indices."""
        return self.index(coord)

    def __iter__(self) -> Iterator[Coords]:
        """Iterate over all on-site and nearest-neighbor interactions."""
        for index in self.sites():
            yield (index, index)
        for indices in self.bonds():
            yield indices

    def index(self, coord: Coord) -> Index:
        """Convert a 3D site coordinate to a 1D index."""
        raise NotImplementedError

    def sites(self) -> Iterator[Coord]:
        """Iterate over all atomic sites in the lattice."""
        raise NotImplementedError

    def bonds(self) -> Iterator[Coords]:
        """Iterate over all atomic bonds in the lattice."""
        raise NotImplementedError


class CubicLattice(Lattice):
    """Concrete representation of a primitive cubic lattice."""

    def index(self, coord: Coord) -> Index:
        """Convert a 3D site coordinate to a 1D index."""

        return coord[2] + coord[1] * self.shape[2] + coord[0] * self.shape[1] * self.shape[2]

    def sites(self) -> Iterator[Coord]:
        """Iterate over all atomic sites in the lattice."""

        for x in range(self.shape[0]):
            for y in range(self.shape[1]):
                for z in range(self.shape[2]):
                    yield (x, y, z)

    def bonds(self, axis: Optional[int] = None) -> Iterator[Coords]:
        """Iterate over all atomic bonds in the lattice.

        The `axis` argument allows iterating over bonds along only one
        cardinal axis, where `axis=0` corresponds to the x-axis, etc.
        """

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
