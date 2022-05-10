from collections import namedtuple
from typing import Any, Callable, Iterator, Optional

from numpy.typing import ArrayLike
from numpy.typing import NDArray as Array
from scipy.sparse import bsr_matrix as Sparse

# Data types for working with Lattice coordinates.
Index = int
Coord = tuple[int, int, int]
Indices = tuple[Index, Index]
Coords = tuple[Coord, Coord]

# Named tuples for important return types.
SparseLike = namedtuple("SparseLike", ["data", "indices", "indptr"])
SpectralTuple = namedtuple("SpectralTuple", ["spectral", "energy", "weight"])
