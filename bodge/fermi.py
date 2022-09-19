import numpy as np
from tqdm import tqdm

from .hamiltonian import *
from .math import *
from .stdio import *
from .typing import *


class FermiMatrix:
    def __init__(self, hamiltonian: Hamiltonian, order: int):
        # Store initialization arguments.
        self.hamiltonian: Hamiltonian = hamiltonian
        self.lattice: Lattice = hamiltonian.lattice
        self.order: int = order

        # Storage for the Fermi matrix.
        self.matrix: bsr_matrix

    def __call__(self, temperature: float, radius: Optional[int] = None):
        """Calculate the Fermi matrix at a given temperature."""
        log(self, "Performing Fermi-Chebyshev expansion")

        # Hamiltonian and related matrices.
        H = self.hamiltonian.matrix
        S = self.hamiltonian.struct
        I = self.hamiltonian.identity

        # Fermi function.
        def f(x):
            return 1 / (1 + np.exp(x / temperature))

        # Generators for coefficients and matrices.
        Ts = cheb_poly(H, I, self.order, radius)
        fs = cheb_coeff(f, self.order)
        gs = cheb_kern(self.order)

        # Initialize the Fermi matrix skeleton.
        self.matrix = bsr_matrix(H.shape, blocksize=H.blocksize, dtype=H.dtype)

        # Perform kernel polynomial expansion.
        # TODO: Check adjustments for entropy.
        for f, g, T in tqdm(zip(fs, gs, Ts), total=self.order):
            self.matrix += (f * g * T).multiply(S)

        return self.matrix

    def index(self, row: Coord, col: Coord) -> Index:
        """Sparse matrix index corresponding to block (row, col)."""
        indices, indptr = self.matrix.indices, self.matrix.indptr

        i, j = self.lattice[row], self.lattice[col]
        js = indices[indptr[i] : indptr[i + 1]]
        k = indptr[i] + np.where(js == j)

        return Index(k)
