from typing import Union
from beartype import beartype as typecheck
from beartype.typing import Any, Callable, Iterator, NamedTuple, Optional
from numpy.typing import ArrayLike, DTypeLike
from numpy.typing import NDArray as DenseArray
from scipy.sparse import spmatrix as SparseArray

# Data types for working with Lattice coordinates.
Index = int
Coord = tuple[int, int, int]
Indices = tuple[Index, Index]
Coords = tuple[Coord, Coord]

# Data types for general matrices.
Array = Union[SparseArray, DenseArray]