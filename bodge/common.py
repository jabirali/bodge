"""Common imports and type definitions for the whole project."""

# Core numerical libraries.
import math

import multiprocess as mp
import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as sa

# Enable runtime type checks.
from beartype import beartype as typecheck
from beartype.typing import Any, Callable, Iterator, Optional, Union

# Miscellaneous conveniences.
from tqdm import tqdm, trange

# Data types for working with Lattice coordinates.
Index = int
Coord = tuple[int, int, int]
Indices = tuple[Index, Index]
Coords = tuple[Coord, Coord]

# Data types for working with various matrix formats.
Matrix = npt.NDArray[np.complex128]
SpMatrix = sp.spmatrix
CooMatrix = sp.coo_matrix
DiaMatrix = sp.dia_matrix
BsrMatrix = sp.bsr_matrix
CsrMatrix = sp.csr_matrix
CscMatrix = sp.csc_matrix

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
