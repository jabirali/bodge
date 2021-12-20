"""Unit tests for the utility functions defined in `utils.py`."""

import pytest

from numpy import prod

from pybdg.utils import *


class TestUtils:
    def test_coord2index(self):
        """Test conversion of (iˣ, iʸ, iᶻ) tuples to a single index."""
        # Test 1D indices.
        dims = (10, 0, 0)
        assert coord2index((0, 0, 0), dims) == 0
        assert coord2index((9, 0, 0), dims) == prod(dims[:1]) - 1

        # Test 2D indices.
        dims = (2, 5, 0)
        assert coord2index((0, 0, 0), dims) == 0
        assert coord2index((1, 0, 0), dims) == 1
        assert coord2index((0, 1, 0), dims) == 2
        assert coord2index((1, 1, 0), dims) == 3
        assert coord2index((1, 4, 0), dims) == prod(dims[:2]) - 1

        # Test 3D indices.
        dims = (2, 3, 5)
        assert coord2index((0, 0, 0), dims) == 0
        assert coord2index((1, 0, 0), dims) == 1
        assert coord2index((0, 1, 0), dims) == 2
        assert coord2index((1, 1, 0), dims) == 3
        assert coord2index((0, 2, 0), dims) == 4
        assert coord2index((1, 2, 0), dims) == 5
        assert coord2index((0, 0, 1), dims) == 6
        assert coord2index((1, 2, 4), dims) == prod(dims[:3]) - 1

    def test_index2coord(self):
        """Test that `coord2index` can be successfully inverted."""
        dims = (2, 3, 5)
        for iᶻ in range(dims[2]):
            for iʸ in range(dims[1]):
                for iˣ in range(dims[0]):
                    inds = (iˣ, iʸ, iᶻ)
                    assert index2coord(coord2index(inds, dims), dims) == inds
