from .common import *


class Lattice:
    """API for working with generic three-dimensional lattices.

    This is an abstract class which defines an API for iterating over all the
    sites (atoms) and bonds (nearest neighbors) in a lattice. In the language
    of graph theory, this class lets us traverse all the nodes and links of a
    simple graph. The actual graph traversal must be defined by subclassing
    `Lattice` and implementing all methods that raise `NotImplementedError`.

    Note that you are free to implement optional arguments to these methods.
    For instance, it may be useful to iterate over one sublattice at a time
    when calling `.sites` on a honeycomb lattice, or to iterate over the x-
    and y-axes separately when calling `.bonds` on a rectangular lattice.
    However, it must be possible to call both methods without additional
    arguments to traverse all sites and bonds in the lattice, respectively.
    """

    @typecheck
    def __init__(self, shape: Coord):
        # Create an abstract class.
        if self.__class__.__name__ == "Lattice":
            raise ValueError("This class is not intended to be instantiated directly.")

        # Number of atoms per dimension.
        self.shape: Coord = shape

        # Number of atoms in the lattice.
        self.size: Index = int(np.prod(shape))

        # Dimensionality of the lattice.
        self.dim: int = sum([1 for x in self.shape if x > 1])

    @typecheck
    def __getitem__(self, coord: Coord) -> Index:
        """Syntactic sugar for converting coordinates into indices."""
        return self.index(coord)

    @typecheck
    def __iter__(self) -> Iterator[Coords]:
        """Iterate over all on-site and nearest-neighbor interactions."""
        for index in self.sites():
            yield (index, index)
        for indices in self.bonds():
            yield indices

    @typecheck
    def index(self, coord: Coord) -> Index:
        """Convert a 3D site coordinate to a 1D index."""
        raise NotImplementedError

    @typecheck
    def sites(self) -> Iterator[Coord]:
        """Iterate over all atomic sites in the lattice."""
        raise NotImplementedError

    @typecheck
    def bonds(self) -> Iterator[Coords]:
        """Iterate over all atomic bonds in the lattice."""
        raise NotImplementedError

class CubicLattice(Lattice):
    """Concrete representation of a primitive cubic lattice."""

    @typecheck
    def index(self, coord: Coord) -> Index:
        """Convert a 3D site coordinate to a 1D index."""
        return coord[2] + coord[1] * self.shape[2] + coord[0] * self.shape[1] * self.shape[2]

    @typecheck
    def sites(self) -> Iterator[Coord]:
        """Iterate over all atomic sites in the lattice."""
        for x in range(self.shape[0]):
            for y in range(self.shape[1]):
                for z in range(self.shape[2]):
                    yield (x, y, z)

    @typecheck
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
                        yield (x + 1, y, z), (x, y, z)
        elif axis == 1:
            # Neighbors along y-axis.
            for x in range(self.shape[0]):
                for y in range(self.shape[1] - 1):
                    for z in range(self.shape[2]):
                        yield (x, y, z), (x, y + 1, z)
                        yield (x, y + 1, z), (x, y, z)
        elif axis == 2:
            # Neighbors along z-axis.
            for x in range(self.shape[0]):
                for y in range(self.shape[1]):
                    for z in range(self.shape[2] - 1):
                        yield (x, y, z), (x, y, z + 1)
                        yield (x, y, z + 1), (x, y, z)
        else:
            raise ValueError("No such axis")

    def edges(self, axis: int):
        """Iterate over pairs of atoms at opposite lattice edges.

        This is useful if you need periodic boundary conditions. For example,
        to specify periodic boundary conditions along the x-axis, iterate over
        `.edges(axis=0)` and define hopping amplitudes between these atoms.
        """
        Lx, Ly, Lz = self.shape[0], self.shape[1], self.shape[2]
        if axis == 0:
            # Edges at x=0 and x=Lx-1.
            for y in range(Ly):
                for z in range(Lz):
                    yield (0, y, z), (Lx-1, y, z)
                    yield (Lx-1, y, z), (0, y, z)
        elif axis == 1:
            # Edges at y=0 and y=Ly-1.
            for x in range(Lx):
                for z in range(Lz):
                    yield (x, 0, z), (x, Ly-1, z)
                    yield (x, Ly-1, z), (x, 0, z)
        elif axis == 2:
            # Edges at z=0 and z=Lz-1.
            for x in range(Lx):
                for y in range(Ly):
                    yield (x, y, 0), (x, y, Lz-1)
                    yield (x, y, Lz-1), (x, y, 0)
        else:
            raise ValueError("No such axis")