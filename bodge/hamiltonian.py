from .common import *
from .lattice import Lattice


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

        # Initialize the most general 4N×4N Hamiltonian for this lattice as a
        # sparse matrix. The COO format is most efficient for constructing the
        # matrix, but the BSR format is more computationally efficient later.
        sites = sum(1 for _ in lattice.sites())
        bonds = sum(2 for _ in lattice.bonds())
        edges = sum(2 for _ in lattice.edges())
        size = sites + bonds + edges

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

        skeleton = CooMatrix((data, (rows, cols)), shape=self.shape).tobsr((4, 4))

        # Save an integer matrix that encodes the structure of the Hamiltonian,
        # i.e. any potentially present element in the matrix is indicated by 1.
        # This can be used to instantiate new matrices with the same structure.
        self.mask: BsrMatrix = BsrMatrix(skeleton, dtype=np.int8)
        self.mask.data[...] = 1

        # Save an identity matrix of the same dimension as the Hamiltonian.
        self.identity: DiaMatrix = sp.identity(self.shape[0], "int8")

        # Save a complex matrix that encodes the Hamiltonian matrix itself.
        # Each element is set to zero and must later be populated for use.
        self.matrix: BsrMatrix = BsrMatrix(skeleton, dtype=np.complex128)
        self.matrix.data[...] = 0

        # Simplify direct access to the underlying data structure.
        self.data: Matrix = self.matrix.data

        # Storage for any Hubbard-type potentials on the lattice.
        self.pot: dict[Coords, float] = {}
        for i in self.lattice:
            self.pot[i] = 0

    @typecheck
    def __enter__(
        self,
    ) -> tuple[dict[Coords, Matrix], dict[Coords, Matrix], dict[Coords, float]]:
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
        - Verifying that the constructed Hamiltonian is actually Hermitian.
        """
        # Process hopping terms: H[i, j].
        for (i, j), val in self.hopp.items():
            # Find this matrix block.
            k = self.index(i, j)

            # Update respecting electron-hole symmetry.
            self.data[k, 0:2, 0:2] = +val
            self.data[k, 2:4, 2:4] = -val.conj()

        # Process pairing terms: Δ[i, j].
        for (i, j), val in self.pair.items():
            # Find this matrix block.
            k1 = self.index(i, j)
            k2 = self.index(j, i)

            # Update respecting Hermitian symmetry.
            self.data[k1, 0:2, 2:4] = +val
            self.data[k2, 2:4, 0:2] = +val.T.conj()

        # Verify that the matrix is Hermitian.
        if np.max(self.matrix - self.matrix.getH()) > 1e-6:
            raise RuntimeError("The constructed Hamiltonian is not Hermitian!")

        # Reset accessors.
        del self.hopp
        del self.pair

    @typecheck
    def __call__(self, format="csr") -> Union[tuple[SpMatrix, SpMatrix, SpMatrix], Matrix]:
        """Return a specific matrix representation of the Hamiltonian.

        If `format = "dense"`, then the Hamiltonian is returned as a dense
        matrix (i.e. `np.array`). If `format = "bsr" | "csr" | "csc"`, then
        three sparse matrices (i.e. `scipy.sparse.spmatrix`) are returned.
        The first is the Hamiltonian itself. The second is a mask indicating
        every term that *could* be present in a general Hamiltonian with
        nearest-neighbor interactions. The last is an identity matrix.
        """
        # Get relevant stored fields.
        H = self.matrix
        M = self.mask
        I = self.identity

        # Transform as needed and eliminate zeros if possible.
        if format == "bsr":
            H = H.copy()
            H.eliminate_zeros()
            M = M.copy()
            I = I.tobsr(H.blocksize)

            return H, M, I
        elif format == "csr":
            H = H.tocsr()
            H.eliminate_zeros()
            M = M.tocsr()
            I = I.tocsr()
        elif format == "csc":
            H = H.tocsc()
            H.eliminate_zeros()
            M = M.tocsc()
            I = I.tocsc()
        elif format == "dense":
            return H.todense()
        else:
            raise RuntimeError("Unsupported matrix format")

        return H, M, I

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
