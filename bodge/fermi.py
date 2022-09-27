import numpy as np
from tqdm import tqdm

from .hamiltonian import *
from .math import *
from .stdio import *
from .typing import *


class FermiMatrix:
    """Representation of the Fermi operator of a physical system.

    The Fermi matrix is defined by evaluating the Fermi function with the
    Hamiltonian matrix as its argument: F = f(H). In the basis where the
    Hamiltonian is diagonal, this produces F = diag[f(ε_1), ..., f(ε_N)]
    which contains real occupation numbers on its diagonal. When the
    Hamiltonian is expressed in terms of real-space lattice sites, we
    can show that physical observables such as currents and order
    parameters can be read out directly from the matrix elements.

    This class facilitates the computation and use of the Fermi matrix.
    In practice, this is done via a Chebyshev matrix expansion of f(H).
    """

    def __init__(self, hamiltonian: Hamiltonian, order: int):
        # Store initialization arguments.
        self.hamiltonian: Hamiltonian = hamiltonian
        self.lattice: Lattice = hamiltonian.lattice
        self.order: int = order

        # Storage for the Fermi matrix.
        self.matrix: bsr_matrix

        # Accessors for context manager.
        self.hopp: dict[Coords]

    def __call__(self, temperature: float, radius: Optional[int] = None):
        """Calculate the Fermi matrix at a given temperature."""
        log(self, "Fermi-Chebyshev expansion")

        # Hamiltonian and related quantities.
        H = self.hamiltonian.matrix
        S = self.hamiltonian.struct
        I = self.hamiltonian.identity
        Ω = self.hamiltonian.scale

        # Define the Fermi function.
        def f(x):
            return 1 / (1 + np.exp(Ω * x / temperature))

        # Generators for coefficients and matrices.
        Ts = cheb_poly(H, I, self.order, radius)
        fs = cheb_coeff(f, self.order, odd=True)
        gs = cheb_kern(self.order)

        # Initialize the Fermi matrix skeleton.
        self.matrix = bsr_matrix(H.shape, blocksize=H.blocksize, dtype=H.dtype)

        # Perform kernel polynomial expansion.
        # TODO: Check adjustments for entropy.
        for f, g, T in tqdm(zip(fs, gs, Ts), desc=" -> expanding", unit="", total=self.order):
            if f != 0:
                self.matrix += (f * g * T).multiply(S)

        return self

    def __enter__(self):
        """Implement a context manager interface for the class."""
        # Prepare accessors.
        self.hopp = {}
        self.pair = {}

        # Process all relevant terms in the lattice.
        log(self, "Extracting matrix elements")
        for i, j in tqdm(self.lattice, desc=" -> extracting", unit=""):
            # Find the lattice-transposed matrix block. This is useful because
            # the Fermi matrix block F[i, j] contains the reversed ⟨cj^† ci⟩.
            k1 = self.index(j, i)

            # Fill the accessor dictionaries with relevant matrix blocks.
            self.hopp[(i, j)] = self.matrix.data[k1, :2, :2].T
            self.pair[(i, j)] = self.matrix.data[k1, :2, 2:].T

        return self.hopp, self.pair

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up after the context manager."""
        del self.hopp
        del self.pair

    def index(self, row: Coord, col: Coord) -> Index:
        """Sparse matrix index corresponding to block (row, col)."""
        indices, indptr = self.matrix.indices, self.matrix.indptr

        i, j = self.lattice[row], self.lattice[col]
        js = indices[indptr[i] : indptr[i + 1]]
        k = indptr[i] + np.where(js == j)

        return Index(k)
