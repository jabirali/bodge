import h5py
import numpy as np
from scipy.linalg import eigh, inv
from scipy.sparse import bsr_matrix, coo_matrix, identity
from scipy.sparse.linalg import norm, eigsh
import multiprocess as mp

from .lattice import Lattice
from .math import *
from .stdio import *
from .typing import *


class Hamiltonian:
    """Representation of a physical system in the tight-binding limit.

    This class can be used to construct Hamiltonian matrices for condensed
    matter systems with particle-hole and spin degrees of freedom. Instead of
    explicitly constructing the whole matrix, this class allows you to specify
    the minimum number of matrix elements required via a `with` block, and the
    the remainder is autofilled via symmetries. Moreover, it allows you to use
    lattice coordinates instead of matrix indices to fill out these elements.

    Internally, this constructs a block-sparse matrix (BSR format), enabling
    the construction of megadimensional tight-binding systems (>10^6 sites).
    """

    @typecheck
    def __init__(self, lattice: Lattice):
        # Lattice instance used as basis coordinates for the system.
        self.lattice: Lattice = lattice

        # Number of lattice points that the system has. The integers in front of
        # the lattice size are the local degrees of freedom at each lattice site.
        self.shape: Indices = (4 * lattice.size, 4 * lattice.size)

        # Scale factor used to compress the Hamiltonian spectrum to (-1, +1).
        # This must be set to an upper bound for the Hamiltonian spectral radius.
        self.scale: float = 1.0

        # Initialize the most general 4N×4N Hamiltonian for this lattice as a
        # sparse matrix. The COO format is most efficient for constructing the
        # matrix, but the BSR format is more computationally efficient later.
        size = sum(1 for _ in lattice.sites()) + sum(2 for _ in lattice.bonds())

        rows = np.zeros(size, dtype=np.int64)
        cols = np.zeros(size, dtype=np.int64)
        data = np.repeat(np.int8(1), size)

        k = 0
        for ri, rj in lattice:
            i, j = 4 * lattice[ri], 4 * lattice[rj]

            rows[k] = i
            cols[k] = j
            k += 1

            if i != j:
                rows[k] = j
                cols[k] = i
                k += 1

        skeleton = coo_matrix((data, (rows, cols)), shape=self.shape).tobsr((4, 4))

        # Save an integer matrix that encodes the structure of the Hamiltonian,
        # i.e. any potentially present element in the matrix is indicated by 1.
        # This can be used to instantiate new matrices with the same structure.
        self.struct: bsr_matrix = bsr_matrix(skeleton, dtype=np.int8)
        self.struct.data[...] = 1

        # Save a complex matrix that encodes the Hamiltonian matrix itself.
        # Each element is set to zero and must later be populated for use.
        self.matrix: bsr_matrix = bsr_matrix(skeleton, dtype=np.complex128)
        self.matrix.data[...] = 0

        # Simplify direct access to the underlying data structure.
        self.data: Array[np.complex128] = self.matrix.data


        # Storage for any Hubbard-type potentials on the lattice.
        self.pot: dict[Coords, float] = {}

    @typecheck
    def __enter__(self) -> tuple[dict[Coords, Array], dict[Coords, Array], dict[Coords, float]]:
        """Implement a context manager interface for the class.

        This lets us write compact `with` blocks like the below, which is much
        more convenient than having to construct the matrix elements explicitly.

        ```python
        with system as (H, Δ, V):
            H[i, j] = ...
            Δ[i, j] = ...
            V[i, j] = ...
        ```

        Note that the `__exit__` method is responsible for actually transferring
        all the elements of H and Δ to the correct locations in the Hamiltonian.
        """
        # Restore the Hamiltonian energy scale.
        self.data *= self.scale

        # Prepare storage for the context manager.
        self.hopp = {}
        self.pair = {}

        return self.hopp, self.pair, self.pot

    @typecheck
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Implement a context manager interface for the class.

        This part of the implementation takes care of finalizing the Hamiltonian:

        - Transferring elements from the context manager dicts to the sparse matrix;
        - Ensuring that particle-hole and nearest-neighbor symmetries are respected;
        - Verifying that the constructed Hamiltonian is actually Hermitian;
        - Scaling the Hamiltonian to have a spectrum bounded by (-1, +1).
        """
        # Process hopping terms: H[i, j].
        for (i, j), val in self.hopp.items():
            # Find this matrix block.
            k1 = self.index(i, j)

            # Update electron-electron and hole-hole parts.
            self.data[k1, 0:2, 0:2] = +val
            self.data[k1, 2:4, 2:4] = -val.conj()

            # Inverse process for non-diagonal contributions.
            if i != j:
                k2 = self.index(j, i)
                self.data[[[k2]], ...] = np.swapaxes(self.data[[[k1]], ...], 2, 3).conj()

        # Process pairing terms: Δ[i, j].
        for (i, j), val in self.pair.items():
            # Find this matrix block.
            k1 = self.index(i, j)

            # Update electron-hole and hole-electron parts.
            self.data[k1, 0:2, 2:4] = +val
            self.data[k1, 2:4, 0:2] = +val.T.conj()

            # Inverse process for non-diagonal contributions.
            if i != j:
                k2 = self.index(j, i)
                self.data[[[k2]], ...] = np.swapaxes(self.data[[[k1]], ...], 2, 3).conj()

        # Verify that the matrix is Hermitian.
        if np.max(self.matrix - self.matrix.getH()) > 1e-6:
            raise RuntimeError("The constructed Hamiltonian is not Hermitian!")

        # Scale the matrix so all eigenvalues are in (-1, +1). We here use
        # the theorem that the spectral radius is bounded by any matrix norm.
        self.scale: float = norm(self.matrix, 1)
        self.matrix /= self.scale

        # Reset accessors.
        del self.hopp
        del self.pair

    @typecheck
    def index(self, row: Coord, col: Coord) -> Index:
        """Determine the sparse matrix index corresponding to block (row, col).

        This can be used to access `self.data[index, :, :]` when direct
        changes to the encapsulated block-sparse matrix are required.
        """
        indices, indptr = self.matrix.indices, self.matrix.indptr

        i, j = self.lattice[row], self.lattice[col]
        js = indices[indptr[i] : indptr[i + 1]]
        k = indptr[i] + np.where(js == j)

        return Index(k)

    @property
    @typecheck
    def identity(self) -> bsr_matrix:
        """Generate an identity matrix with similar dimensions as the Hamiltonian."""
        return identity(self.shape[1], "int8").tobsr((4, 4))

    @typecheck
    def diagonalize(self) -> tuple[Array, Array]:
        """Calculate the exact eigenstates of the system via direct diagonalization.

        This calculates the eigenvalues and eigenvectors of the system. Due to
        the particle-hole symmetry, only positive eigenvalues are calculated.
        Note that this method is inefficient since it uses dense matrices; it
        is meant as a benchmark, not for actual large-scale calculations.
        """
        # Calculate the relevant eigenvalues and eigenvectors.
        H = self.scale * self.matrix.todense()
        eigval, eigvec = eigh(H, subset_by_value=(0, np.inf))

        # Restructure the eigenvectors to have the format eigvec[n, i, α],
        # where n corresponds to eigenvalue E[n], i is a position index, and
        # α represents the combined particle and spin index {e↑, e↓, h↑, h↓}.
        eigvec = eigvec.T.reshape((eigval.size, -1, 4))

        return eigval, eigvec

    @typecheck
    def eigenvalues(self, method: str) -> Array:
        """Calculate the exact eigenvalues of the system.

        Due to particle-hole symmetry, only positive eigenvalues are calculated.
        Note that this method is inefficient since it uses dense matrices.
        """
        if method == 'dense':
            H = self.scale * self.matrix.todense()
            E = eigh(H, overwrite_a=True, eigvals_only=True, driver='evr')
        elif method == 'sparse':
            H = self.scale * self.matrix
            E = eigsh(H, H.shape[0] - 2, which='SM', tol=1e-8, return_eigenvectors=False,)


        return np.array(sorted(E_k for E_k in E if E_k > 0))

    def spectral(self, energies: ArrayLike, resolution: float = 1e-2) -> list[Array]:
        """Calculate the exact spectral function of the system via direct inversion.

        Note that this method is quite inefficient since it uses dense matrices;
        it is meant as a benchmark, not for actual large-scale calculations.
        """
        # Restore the Hamiltonian scale and switch to dense matrices.
        H = self.scale * self.matrix.todense()
        I = self.identity.todense()

        # The resolution is controlled by the imaginary energy.
        η = self.scale * resolution * 1j

        # Calculate the spectral function via direct inversion.
        spectral = []
        for ω in energies:
            Gᴿ = inv((ω + η) * I - H)
            Gᴬ = inv((ω - η) * I - H)
            A = (Gᴿ - Gᴬ) / (-2j * π)

            spectral.append(A)

        return spectral

    def plot(self, grid: bool = False):
        """Visualize the sparsity structure of the generated matrix."""
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 8))
        plt.spy(self.matrix, markersize=1, marker="o", color="k")
        plt.title("Hamiltonian elements stored in the Block Sparse Row (BSR) representation")
        plt.xticks([])
        plt.yticks([])

        if grid:
            plt.xticks([4 * i - 0.5 for i in range(self.lattice.size)])
            plt.yticks([4 * i - 0.5 for i in range(self.lattice.size)])
            plt.grid()

        ax = plt.gca()
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        plt.tight_layout()
        plt.show()
