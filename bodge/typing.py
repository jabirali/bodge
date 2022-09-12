from beartype import beartype as typecheck
from beartype.typing import Any, Callable, Iterator, NamedTuple, Optional
from numpy.typing import ArrayLike, DTypeLike
from numpy.typing import NDArray as Array
from scipy.sparse import bsr_matrix

# Data types for working with Lattice coordinates.
Index = int
Coord = tuple[int, int, int]
Indices = tuple[Index, Index]
Coords = tuple[Coord, Coord]
