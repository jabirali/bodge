"""Common imports and type definitions for the whole project."""

# Type annotations that are checked at runtime.
from beartype import beartype as typecheck
from beartype.typing import Any, Callable, Iterator, Optional, Union

# NumPy/SciPy array types.
from numpy.typing import ArrayLike
from numpy.typing import NDArray as DenseArray
from scipy.sparse import spmatrix as SparseArray
from scipy.sparse import bsr_matrix, coo_matrix, csr_matrix, dia_matrix, identity, spmatrix

# Data types for working with Lattice coordinates.
Index = int
Coord = tuple[int, int, int]
Indices = tuple[Index, Index]
Coords = tuple[Coord, Coord]

# Data types for general matrices.
Array = Union[SparseArray, DenseArray]


# Numerical imports
import multiprocess as mp
import numpy as np
import scipy.sparse as sps

# Fundamental constants.
π = np.pi

# Pauli matrices used to represent spin.
σ0 = np.array([[+1, 0], [0, +1]], dtype=np.complex128)
σ1 = np.array([[0, +1], [+1, 0]], dtype=np.complex128)
σ2 = np.array([[0, -1j], [+1j, 0]], dtype=np.complex128)
σ3 = np.array([[+1, 0], [0, -1]], dtype=np.complex128)

σ = np.stack([σ1, σ2, σ3])

# Compact notation for imaginary versions.
jσ0 = 1j * σ0
jσ1 = 1j * σ1
jσ2 = 1j * σ2
jσ3 = 1j * σ3

jσ = np.stack([jσ1, jσ2, jσ3])


# Progress bars
from tqdm import tqdm, trange
