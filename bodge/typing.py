from typing import Any, Callable, Iterator, NamedTuple, Optional

from numpy.typing import ArrayLike, DTypeLike
from numpy.typing import NDArray as Array
from scipy.sparse import bsr_matrix as Sparse
from typeguard import typechecked

# Data types for working with Lattice coordinates.
Index = int
Coord = tuple[int, int, int]
Indices = tuple[Index, Index]
Coords = tuple[Coord, Coord]

# Named tuples for important return types.
class Spectral(NamedTuple):
    energy: float
    value: Sparse
