import numpy as np

from .hamiltonian import *
from .math import *
from .stdio import *
from .typing import *


class FermiMatrix:
    """Representation of the Fermi operator of a physical system.

    The Fermi matrix is defined by evaluating the Fermi function with the
    Hamiltonian matrix as its argument: F = f(H). When the Hamiltonian is
    expressed in terms of real-space lattice sites, physical observables such
    as currents and order parameters can be read directly from this matrix.

    This class facilitates the computation and use of the Fermi matrix.
    In practice, this is done via a Chebyshev matrix expansion of f(H),
    followed by an explicit calculation of various traces of this matrix.
    """

    def __init__(self, hamiltonian: Hamiltonian, order: int):
        # Store initialization arguments.
        self.hamiltonian: Hamiltonian = hamiltonian
        self.lattice: Lattice = hamiltonian.lattice
        self.order: int = order

        # Fermi matrix and its accessors.
        self.matrix: bsr_matrix
        self.hopp: dict[Coords, Array]
        self.pair: dict[Coords, Array]

    def __call__(self, temperature: float, radius: Optional[int] = None):
        """Calculate the Fermi matrix at a given temperature."""
        # Reset any pre-existing accessors.
        self.hopp = {}
        self.pair = {}

        # Hamiltonian and related quantities.
        H = self.hamiltonian.matrix
        S = self.hamiltonian.struct
        I = self.hamiltonian.identity
        Ω = self.hamiltonian.scale

        # Define the Fermi function.
        def f(x):
            return (1 - np.tanh((Ω * x) / (2 * temperature))) / 2

        # Perform kernel polynomial expansion.
        # TODO: Check adjustments for entropy, or whether to .multiply(S).
        self.matrix = cheb(f, H, self.order, radius)

        # Simplify the access to the constructed matrix.
        for i, j in self.lattice:
            # Find the lattice-transposed matrix blocks. This is useful because
            # the Fermi matrix block F[i, j] contains the reversed ⟨cj^† ci⟩.
            k1 = self.index(j, i)
            k2 = self.index(i, j)

            # Fill the accessor dictionaries with spin-resolved blocks.
            if k1 is not None:
                self.hopp[(i, j)] = self.matrix.data[k1, :2, :2].T
                self.pair[(i, j)] = self.matrix.data[k1, :2, 2:].T

            if k2 is not None:
                self.hopp[(j, i)] = self.matrix.data[k2, :2, :2].T
                self.pair[(j, i)] = self.matrix.data[k2, :2, 2:].T

        return self

    def index(self, row: Coord, col: Coord) -> Optional[Index]:
        """Sparse matrix index corresponding to block (row, col)."""
        indices, indptr = self.matrix.indices, self.matrix.indptr

        i, j = self.lattice[row], self.lattice[col]
        js = indices[indptr[i] : indptr[i + 1]]
        k = indptr[i] + np.where(js == j)

        return Index(k) if k.size > 0 else None

    def order_swave(self):
        """Calculate the s-wave singlet order parameter."""
        # TODO: Stencil for p-wave, d-wave, etc.
        V = self.hamiltonian.pot
        Ω = self.hamiltonian.scale
        Δ = np.zeros(self.lattice.shape, dtype=np.complex128)
        for i in self.lattice.sites():
            Δ[i] = (V[i, i] / 2) * np.trace(self.pair[i, i] @ jσ2)

        return Δ

    def current_elec(self, axis):
        """Calculate the electric current on the lattice."""
        # TODO: Factors t_ij
        Ω = self.hamiltonian.scale
        J = np.zeros(self.lattice.shape, dtype=np.float64)
        for i, j in self.lattice.bonds(axis):
            J[i] = (Ω / 2) * np.imag(np.trace(self.hopp[i, j] - self.hopp[j, i]))

        return J

    def entropy(self):
        pass

    def free_energy(self):
        pass
