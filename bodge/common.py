"""Common imports and definitions used throughout the project."""

# Common imports.
import numpy as np
import numpy.typing as npt
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from beartype import beartype as typecheck
from beartype.typing import Callable, Iterator

# Data types for working with Lattice coordinates.
Index = int
Coord = tuple[int, int, int]
Indices = tuple[Index, Index]
Coords = tuple[Coord, Coord]

# Data types for working with various matrix formats.
Matrix = npt.NDArray[np.float64] | npt.NDArray[np.complex128]
CooMatrix = sp.coo_matrix
DiaMatrix = sp.dia_matrix
BsrMatrix = sp.bsr_matrix
CsrMatrix = sp.csr_matrix
CscMatrix = sp.csc_matrix
SpMatrix = sp.spmatrix

# Fundamental constants.
π = np.pi

# Pauli matrices used to represent spin.
σ0: Matrix = np.array([[+1, 0], [0, +1]], dtype=np.complex128)
σ1: Matrix = np.array([[0, +1], [+1, 0]], dtype=np.complex128)
σ2: Matrix = np.array([[0, -1j], [+1j, 0]], dtype=np.complex128)
σ3: Matrix = np.array([[+1, 0], [0, -1]], dtype=np.complex128)

σ = np.stack([σ1, σ2, σ3])

# Compact notation for imaginary versions.
jσ0: Matrix = 1j * σ0
jσ1: Matrix = 1j * σ1
jσ2: Matrix = 1j * σ2
jσ3: Matrix = 1j * σ3

jσ = np.stack([jσ1, jσ2, jσ3])

# ASCII alternatives.
pi = π

sigma0 = σ0
sigma1 = σ1
sigma2 = σ2
sigma3 = σ3

sigma = σ

jsigma0 = jσ0
jsigma1 = jσ1
jsigma2 = jσ2
jsigma3 = jσ3

jsigma = jσ
