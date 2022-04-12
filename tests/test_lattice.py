from bodge.lattice import *


class TestCubic:
    def test_sites(self):
        lat = Cube((3, 5, 7))
        for ind, site in enumerate(lat.sites()):
            # Verify that indexing is contiguous.
            assert ind == lat[site]

            # Verify that index bounds are satisfied.
            assert site[0] >= 0 and site[0] < lat.shape[0]
            assert site[1] >= 0 and site[1] < lat.shape[1]
            assert site[2] >= 0 and site[2] < lat.shape[2]

        # Verify that number of elements is correct.
        assert ind == 3 * 5 * 7 - 1

    def test_neighbors(self):
        lat = Cube((2, 3, 5))
        for (x1, y1, z1), (x2, y2, z2) in lat.neighbors(axis=0):
            # Verify neighbors along the x-axis.
            assert x2 == x1 + 1 and y2 == y1 and z2 == z1
        for (x1, y1, z1), (x2, y2, z2) in lat.neighbors(axis=1):
            # Verify neighbors along the y-axis.
            assert x2 == x1 and y2 == y1 + 1 and z2 == z1
        for (x1, y1, z1), (x2, y2, z2) in lat.neighbors(axis=2):
            # Verify neighbors along the z-axis.
            assert x2 == x1 and y2 == y1 and z2 == z1 + 1

        # Verify that the number of neighbors is correct.
        len([*lat.neighbors()]) == 2 * 3 * 5 - 2 * 3 - 3 * 5 - 5 * 2
