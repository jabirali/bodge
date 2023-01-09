from .chebyshev import cheb
from .common import *
from .hamiltonian import Hamiltonian
from .lattice import Lattice


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
        self.scale: float = 1.0

        # Fermi matrix and its accessors.
        self.matrix: BsrMatrix
        self.hopp: dict[Coords, Matrix]
        self.pair: dict[Coords, Matrix]

    def __call__(self, temperature: float):
        """Calculate the Fermi matrix at a given temperature."""
        # Reset any pre-existing accessors.
        self.hopp = {}
        self.pair = {}

        # Hamiltonian and related quantities.
        H, M, I = self.hamiltonian(format="bsr")

        # Scale the matrix so all eigenvalues are in (-1, +1). We here use
        # the theorem that the spectral radius is bounded by any matrix norm.
        self.scale = sa.norm(H, 1)
        H /= self.scale

        # Define the Fermi function.
        def fermi(x):
            return (1 - np.tanh((self.scale * x) / (2 * temperature))) / 2

        def odd(n):
            return n == 0 or n % 2 == 1

        # Perform kernel polynomial expansion.
        # TODO: Check adjustments for entropy, or whether to .multiply(M).
        self.matrix = cheb(fermi, H, M, self.order, filter=odd).tobsr(H.blocksize)

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
        Ω = self.scale
        Δ = np.zeros(self.lattice.shape, dtype=np.complex128)
        for i in self.lattice.sites():
            if (i, i) in V:
                # TODO: DefaultDict or actual matrix
                Δ[i] = (V[i, i] / 2) * np.trace(self.pair[i, i] @ jσ2)

        return Δ

    def current_elec(self, axis):
        """Calculate the electric current on the lattice.

        TODO: Complex hopping amplitudes t_ij if gauge fields exist.
        TODO: Consider going back to bond currents over site currents,
        since those ones behave a bit less weird at vacuum interfaces.
        """
        Ω = self.scale
        J = np.zeros(self.lattice.shape, dtype=np.float64)
        for i, j in self.lattice.bonds(axis):
            try:
                J[i] += (
                    (Ω / 2)
                    * (j[axis] - i[axis])
                    * np.imag(np.trace(self.hopp[i, j] - self.hopp[j, i]))
                )
            except KeyError:
                pass

        return J
