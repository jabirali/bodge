from .common import *


class Lattice:
    """Base class that represents a general atomic lattice (in 1D, 2D, or 3D).

    This abstract class defines a general interface for iterating over all
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
    This is assumed by the class `Hamiltonian` which is built on `Lattice`.
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
        for indices in self.edges():
            yield indices

    @typecheck
    def __repr__(self) -> str:
        """Representation of the object for `print()`."""
        return self.__class__.__name__ + str(self.shape)

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
        """Iterate over all atomic bonds in the lattice.

        The intended usage is that `for i, j in lattice.bonds()` should
        yield all nearest-neighbor sites (i, j) in the lattice.
        """
        raise NotImplementedError

    @typecheck
    def edges(self) -> Iterator[Coords]:
        """Iterate over pairs of edges in the lattice.

        The intended usage is that `for i, j in lattice.edges()` should yield
        sites i and j on *opposite* edges of a system. Thus, hopping terms
        between such sites should result in periodic boundary conditions.
        """
        raise NotImplementedError


class CubicLattice(Lattice):
    """Concrete representation of a primitive cubic lattice.

    The constructor takes as its argument the number of lattice sites along each
    direction in a 3D Euclidean space. So a 10x10x10 cubic lattice is given by:

    >>> lattice = CubicLattice((10, 10, 10))

    The same class can be used to model square lattices or rectangular lattices
    as well. To e.g. construct an 30x30 square lattice, simply do the following:

    >>> lattice = CubicLattice((30, 30, 1))
    """

    @typecheck
    def index(self, coord: Coord) -> Index:
        """Convert a 3D site coordinate to a 1D index."""
        for i in range(3):
            if coord[i] < 0 or coord[i] >= self.shape[i]:
                raise ValueError(f"Coordinate {coord} out of bounds")
        else:
            return coord[2] + coord[1] * self.shape[2] + coord[0] * self.shape[1] * self.shape[2]

    @typecheck
    def sites(self) -> Iterator[Coord]:
        """Iterate over all atomic sites in the lattice.

        Thus, e.g. `for i in lattice.sites()` iterates over all lattice sites.
        """
        for x in range(self.shape[0]):
            for y in range(self.shape[1]):
                for z in range(self.shape[2]):
                    yield (x, y, z)

    @typecheck
    def bonds(self, axis: int | None = None) -> Iterator[Coords]:
        """Iterate over all atomic bonds in the lattice.

        The `axis` argument allows iterating over bonds along only one
        cardinal axis. For example, `for i, j in lattice.bonds(axis=0)`
        would iterate over nearest neighbors connected along the x-axis.

        If `axis` is not set, we iterate over bonds in all directions.
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

    @typecheck
    def edges(self, axis: int | None = None) -> Iterator[Coords]:
        """Iterate over pairs of atoms at opposite lattice edges.

        For instance, if you want periodic boundary conditions along the
        x-axis, you can use `for i, j in lattice.edges(axis=0)` to find the
        lattice sites (i, j) that you should connect via hopping terms.

        If `axis` is not set, we iterate over edges in all directions.
        """
        Lx, Ly, Lz = self.shape[0], self.shape[1], self.shape[2]
        if axis is None:
            # Edges along all axes.
            yield from self.edges(axis=2)
            yield from self.edges(axis=1)
            yield from self.edges(axis=0)
        elif axis == 0:
            # Edges at x=0 and x=Lx-1.
            for y in range(Ly):
                for z in range(Lz):
                    yield (0, y, z), (Lx - 1, y, z)
                    yield (Lx - 1, y, z), (0, y, z)
        elif axis == 1:
            # Edges at y=0 and y=Ly-1.
            for x in range(Lx):
                for z in range(Lz):
                    yield (x, 0, z), (x, Ly - 1, z)
                    yield (x, Ly - 1, z), (x, 0, z)
        elif axis == 2:
            # Edges at z=0 and z=Lz-1.
            for x in range(Lx):
                for y in range(Ly):
                    yield (x, y, 0), (x, y, Lz - 1)
                    yield (x, y, Lz - 1), (x, y, 0)
        else:
            raise ValueError("No such axis")
