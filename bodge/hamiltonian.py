from .common import *
from .lattice import Lattice


class Hamiltonian:
    """Tight-binding representation of a physical system.

    This class can be used to construct Hamiltonian matrices for condensed
    matter systems with particle-hole and spin degrees of freedom. Instead of
    explicitly constructing the whole matrix, this class allows you to specify
    the minimum number of matrix elements required via a `with` block, and the
    the remainder is autofilled via symmetries. Moreover, it allows you to use
    lattice coordinates instead of matrix indices to fill out these elements.

    Internally, this constructs a block-sparse matrix (BSR format), enabling
    the construction of megadimensional tight-binding systems (>10^6 sites).
    However, you can use the `__call__` method to export the result to other
    common sparse matrix formats (e.g. CSR) or a dense matrix (NumPy array).

    For examples of how to use this class, see the bundled documentation.
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

        # Save a complex matrix that encodes the Hamiltonian matrix itself.
        # Each element is set to zero and must later be populated for use.
        self.matrix: BsrMatrix = BsrMatrix(skeleton, dtype=np.complex128)
        self.matrix.data[...] = 0

        # Simplify direct access to the underlying data structure.
        self.data: Matrix = self.matrix.data

    @typecheck
    def __enter__(self) -> tuple[dict[Coords, Matrix], dict[Coords, Matrix]]:
        """Implement a context manager interface for the class.

        This lets us write compact `with` blocks like the below, which is much
        more convenient than having to construct the matrix elements explicitly.

        ```python
        with system as (H, Δ):
            H[i, j] = ...
            Δ[i, j] = ...
        ```

        Note that the `__exit__` method is responsible for actually transferring
        all the elements of H and Δ to the correct locations in the Hamiltonian.
        """
        # Prepare storage for the context manager.
        self.hopp = {}
        self.pair = {}

        return self.hopp, self.pair

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
        if np.max(np.abs(self.matrix - self.matrix.getH())) > 1e-6:
            raise RuntimeError("The constructed Hamiltonian is not Hermitian!")

        # Reset accessors.
        del self.hopp
        del self.pair

    @typecheck
    def __call__(self, format: str = "csr") -> Union[SpMatrix, Matrix]:
        """Return a specific matrix representation of the Hamiltonian.

        If the format is set to "dense", the result is returned as a NumPy
        array. Otherwise, a SciPy sparse matrix is returned instead.
        """
        # Transform the stored matrix into the requested matrix format. Trim
        # any remaining zero entries if a sparse matrix is requested.
        if format == "bsr":
            # NOTE: Don't run eliminate_zeros on the original matrix, as that
            # would preclude adding new elements to the Hamiltonian later.
            H: BsrMatrix = self.matrix.copy()
            H.eliminate_zeros()
        elif format == "csr":
            H: CsrMatrix = self.matrix.tocsr()
            H.eliminate_zeros()
        elif format == "csc":
            H: CscMatrix = self.matrix.tocsc()
            H.eliminate_zeros()
        elif format == "dense":
            H: Matrix = self.matrix.todense()
        else:
            raise RuntimeError("Requested matrix format is not yet supported")

        return H

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

        return Index(k[0, 0])


def swave() -> Matrix:
    """Hamiltonian terms for s-wave superconducting order.

    This is mainly implemented for consistency with `pwave` and `dwave`.
    Moreover, since the structure is the same for on-site and extended
    s-wave orders, it is not a function of coordinates but just a matrix.

    See the documentation for usage examples.
    """

    return jσ2


def pwave(desc: str):
    """Hamiltonian terms for p-wave superconducting order.

    You should provide a d-vector [e.g. "(p_x + jp_y) * (e_x + je_y)"] in order
    to construct a p-wave triplet order parameter. The function then returns a
    new function Δ_p(i, j) which lets you evaluate the superconducting order
    parameter for two lattice sites with coordinates i and j. This is useful
    when constructing the Hamiltonian of the system numerically.

    The algorithm implemented here is explained in detail in Sec. II-B in:

        Ouassou et al. PRB 109, 174506 (2024).
        DOI: 10.1103/PhysRevB.109.174506

    See the documentation for usage examples.
    """
    # Basis vectors for spin axes.
    e_x = np.array([[1], [0], [0]])
    e_y = np.array([[0], [1], [0]])
    e_z = np.array([[0], [0], [1]])

    je_x = 1j * e_x
    je_y = 1j * e_y
    je_z = 1j * e_z

    # Basis vectors for momentum.
    p_x = e_x.T
    p_y = e_y.T
    p_z = e_z.T

    jp_x = 1j * p_x
    jp_y = 1j * p_y
    jp_z = 1j * p_z

    # Convert the d-vector expression to a 3x3 numerical matrix.
    D = eval(desc)

    # Construct gap matrix Δ(p) = [d(p)⋅σ] jσ2 = [(D'p) ⋅ σ] jσ2.
    # In practice, we do this by calculating Δ = D'σ jσ2, such
    # that we simply end up with the gap matrix Δ(p) = Δ ⋅ p.
    Δ = np.einsum("kp,kab,bc -> pac", D, σ, jσ2) / 2

    # Function for evaluating Δ(p) on the lattice.
    def Δ_p(i: Coord, j: Coord) -> Matrix:
        δ = np.subtract(j, i)
        return np.einsum("iab,i -> ab", Δ, δ)

    return Δ_p


def dwave():
    """Generate the d-wave superconducting order parameter.

    This function returns a function Δ(i, j) that takes two lattice sites i, j,
    and returns the superconducting order parameter Δ between those two sites.

    We specifically consider the d_{x^2 - y^2} order parameter on a presumably
    square lattice. This means that the order parameter should have a momentum
    structure ~ (p_x^2 - p_y^2)/p_F^2 and spin structure ~ jσ_2 (spin-singlet).
    It might work on non-square lattices as well, but this has to be verified.

    See the documentation for usage examples.
    """

    def Δ_d(i: Coord, j: Coord) -> Matrix:
        δ = np.subtract(j, i)
        Δ_ij = (δ[0] ** 2 - δ[1] ** 2) / (np.sum(δ**2) + 1e-16)

        return Δ_ij * jσ2

    return Δ_d