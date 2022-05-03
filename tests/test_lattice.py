from pytest import raises

from bodge.lattice import *


def test_super():
    # Superclass should not be constructable.
    with raises(ValueError):
        lat = Lattice((1, 1, 1))


def test_sites():
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


def test_bonds():
    lat = CubicLattice((2, 3, 5))
    for (x1, y1, z1), (x2, y2, z2) in lat.bonds(axis=0):
        # Verify neighbors along the x-axis.
        assert x2 == x1 + 1 and y2 == y1 and z2 == z1
    for (x1, y1, z1), (x2, y2, z2) in lat.bonds(axis=1):
        # Verify neighbors along the y-axis.
        assert x2 == x1 and y2 == y1 + 1 and z2 == z1
    for (x1, y1, z1), (x2, y2, z2) in lat.bonds(axis=2):
        # Verify neighbors along the z-axis.
        assert x2 == x1 and y2 == y1 and z2 == z1 + 1
