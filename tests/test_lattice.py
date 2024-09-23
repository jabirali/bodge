"""Unit tests for the `Lattice` class and its derivatives.

In this set of tests, we mainly try to verify that the implementation
is mathematically correct. For more physically minded tests, please
see the integration tests in `test_physics.py`.
"""

from pytest import raises

from bodge.common import *
from bodge.lattice import *


def test_abc():
    """Test that `Lattice` is an "abstract base class"."""
    # Abstract base class should not be constructable.
    with raises(ValueError):
        lat = Lattice((1, 1, 1))

    # Derived classes are constructable but must override other methods to work.
    class MyLattice(Lattice):
        pass

    lat = MyLattice((1, 2, 3))
    with raises(NotImplementedError):
        lat[(0, 0, 0)]
    with raises(NotImplementedError):
        lat.sites()
    with raises(NotImplementedError):
        lat.bonds()
    with raises(NotImplementedError):
        lat.edges()

    # Derived classes should have a readable format.
    assert str(lat) == "MyLattice(1, 2, 3)"


def test_cubic_sites():
    """Test the iteration over sites in a cubic lattice."""
    lat = CubicLattice((3, 5, 7))
    tot = 0
    for ind, site in enumerate(lat.sites()):
        tot += 1

        # Verify that indexing is contiguous.
        assert ind == lat[site]

        # Verify that index bounds are satisfied.
        assert site[0] >= 0 and site[0] < lat.shape[0]
        assert site[1] >= 0 and site[1] < lat.shape[1]
        assert site[2] >= 0 and site[2] < lat.shape[2]

    # Verify that number of elements is correct.
    assert tot == 3 * 5 * 7

    # Verify that we get out-of-bounds errors.
    with raises(ValueError):
        lat[(-1, 0, 0)]
    with raises(ValueError):
        lat[(0, -1, 0)]
    with raises(ValueError):
        lat[(0, 0, -1)]
    with raises(ValueError):
        lat[(3, 0, 0)]
    with raises(ValueError):
        lat[(0, 5, 0)]
    with raises(ValueError):
        lat[(0, 0, 7)]


def test_cubic_bonds():
    """Test the iteration over bonds in a cubic lattice."""
    tot = 0
    lat = CubicLattice((2, 3, 5))

    # Verify neighbor coordinates along each axis.
    for (x1, y1, z1), (x2, y2, z2) in lat.bonds(axis=0):
        tot += 1
        assert (x2 == x1 + 1 or x2 == x1 - 1) and y2 == y1 and z2 == z1
    for (x1, y1, z1), (x2, y2, z2) in lat.bonds(axis=1):
        tot += 1
        assert x2 == x1 and (y2 == y1 + 1 or y2 == y1 - 1) and z2 == z1
    for (x1, y1, z1), (x2, y2, z2) in lat.bonds(axis=2):
        tot += 1
        assert x2 == x1 and y2 == y1 and (z2 == z1 + 1 or z2 == z1 - 1)

    # Verify the total number of nearest-neighbor pairings. Note the prefactor 2,
    # since "forwards" and "backwards" pairings are both counted here.
    assert tot == 2 * ((2 - 1) * 3 * 5 + 2 * (3 - 1) * 5 + 2 * 3 * (5 - 1))

    # Verify that the bounds checks works.
    with raises(ValueError):
        for i, j in lat.bonds(axis=3):
            print(i, j)


def test_cubic_edges():
    """Test the iteration over edges in a cubic lattice."""
    tot = 0
    lat = CubicLattice((2, 3, 5))

    # Verify that each coordinate is along an edge.
    for (x1, y1, z1), (x2, y2, z2) in lat.edges(axis=0):
        tot += 1
        assert x1 == 0 or x2 == 0
        assert x1 == 1 or x2 == 1
    for (x1, y1, z1), (x2, y2, z2) in lat.edges(axis=1):
        tot += 1
        assert y1 == 0 or y2 == 0
        assert y1 == 2 or y2 == 2
    for (x1, y1, z1), (x2, y2, z2) in lat.edges(axis=2):
        tot += 1
        assert z1 == 0 or z2 == 0
        assert z1 == 4 or z2 == 4

    # Verify that the number of edge sites is correct.
    assert tot == 2 * ((2 * 3) + (3 * 5) + (5 * 2))

    # Verify that the bounds checks works.
    with raises(ValueError):
        for i, j in lat.edges(axis=3):
            print(i, j)
