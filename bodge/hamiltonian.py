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
    def __call__(self, format: str = "csr") -> SpMatrix | Matrix:
        """Return a specific matrix representation of the Hamiltonian.

        If the format is set to "dense", the result is returned as a NumPy
        array. Otherwise, a SciPy sparse matrix is returned instead.
        """
        # Transform the stored matrix into the requested matrix format. Trim
        # any remaining zero entries if a sparse matrix is requested.
        match format:
            case "bsr":
                # NOTE: Don't run eliminate_zeros on the original matrix, as that
                # would preclude adding new elements to the Hamiltonian later.
                H: BsrMatrix = self.matrix.copy()
                H.eliminate_zeros()
            case "csr":
                H: CsrMatrix = self.matrix.tocsr()
                H.eliminate_zeros()
            case "csc":
                H: CscMatrix = self.matrix.tocsc()
                H.eliminate_zeros()
            case "dense":
                H: Matrix = self.matrix.todense()
            case _:
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


def swave() -> Callable:
    """Hamiltonian terms for s-wave superconducting order.

    If you let `σ_s = swave()`, then the on-site pairing term for an
    s-wave superconductor is given by `Δ[i, i] = Δ_s(i) * σ_s(i)`.
    Similarly, extended s-wave order can be obtained by setting
    `Δ[i, j] = Δ_s(i, j) * σ_s(i, j) for nearest-neighbor sites
    `i` and `j`. The prefactors `Δ_s` above can be complex scalar
    functions of the lattice site indices, but can often be set
    constant in non-selfconsistent calculations without currents.
    """

    def σ_s(*_):
        return jσ2

    return σ_s


def pwave(dvector: str) -> Callable:
    """Hamiltonian terms for p-wave superconducting order.

    When calling this function, you must provide a d-vector expression
    as a string [e.g. `σ_p = pwave("(p_x + jp_y) * (e_x + je_y)")`] to
    construct the a p-wave triplet superconducting order parameter.
    The result is a function `σ_p(i, j)` that depend on two
    nearest-neighbor lattice sites `i` and `j`. The superconducting
    pairing is then given by `Δ[i, j] = Δ_p(i, j) * σ_p(i, j)`, where
    `Δ_p` can be any complex scalar function of position `(i + j)/2`.
    It is often sufficient to set the prefactor `Δ_p` to a constant.

    The algorithm implemented here is explained in Sec. II-B in:

    Ouassou et al. PRB 109, 174506 (2024).
    DOI: 10.1103/PhysRevB.109.174506
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
    D = eval(dvector)

    # Construct gap matrix Δ(p) = [d(p)⋅σ] jσ2 = [(D'p) ⋅ σ] jσ2.
    # In practice, we do this by calculating Δ = D'σ jσ2, such
    # that we simply end up with the gap matrix Δ(p) = Δ ⋅ p.
    Δ = np.einsum("kp,kab,bc -> pac", D, σ, jσ2) / 2

    # Function for evaluating Δ(p) on the lattice.
    def σ_p(i: Coord, j: Coord) -> Matrix:
        δ = np.subtract(j, i)
        return np.einsum("iab,i -> ab", Δ, δ)

    return σ_p


def dwave() -> Callable:
    """Generate the d-wave superconducting order parameter.

    This function returns a function `σ_d(i, j)` that takes two
    nearest-neighbor lattice sites `i` and `j` as arguments. The
    d-wave pairing is given by `Δ[i, j] = Δ_d(i, j) * σ_d(i, j)`,
    where the prefactor `Δ_d` can be any complex scalar function
    of position `(i+j)/2`. Often, it can be set to a constant.

    We specifically consider the d_{x^2 - y^2} order parameter on a
    presumably square lattice. This means that the order parameter
    should have a momentum structure ~ (p_x^2 - p_y^2)/p_F^2 and spin
    structure ~ jσ_2 (spin-singlet). It might work on some non-square
    lattices as well, but this has not been checked by the author.
    """

    def σ_d(i: Coord, j: Coord) -> Matrix:
        δ = np.subtract(j, i)
        Δ_ij = (δ[0] ** 2 - δ[1] ** 2) / (np.sum(δ**2) + 1e-16)

        return Δ_ij * jσ2

    return σ_d


def ssd(system: Hamiltonian) -> Callable:
    """Sine-Squared Deformation of a Hamiltonian on a cubic lattice.

    When you call this function as `φ = ssd(system)`, where `system`
    is an instance of the `Hamiltonian` class, you obtain a function
    `φ(i, j)` that depends on two lattice coordinates `i` and `j`. The
    usage of this function is to include `φ(i, j)` as a prefactor for
    every off-site term `H[i, j]` and `Δ[i, j]` in the Hamiltonian,
    and add `φ(i, i)` as a prefactor for every on-site term `H[i, i]`
    and `Δ[i, i]`. This approach is known to reduce finite-size
    effects in real-space simulations, which can be especially
    important in the case of incommensurate order on the lattice.

    For more information about how this approach can be useful, see
    the following reference as well as the references therein:

    Hodt et al. PRB 107, 224427 (2023).
    DOI: 10.1103/PhysRevB.107.224427
    """

    # Define the profile φ(i, j) used in the SSD method.
    def profile(i: Coord, j: Coord):
        # Determine the origin and maximum radius for the system. Since we use
        # corner-centered coordinates, these two values are actually the same.
        # Note however the offset to get a maximum coordinate from a shape.
        R = np.array([(N - 1) for N in system.lattice.shape]) / 2

        # Determine the distance vectors of the provided coordinates
        # i = (i_x, i_y, i_z), j = (j_x, j_y, j_z) from the origin.
        r_i = np.array(i) - R
        r_j = np.array(j) - R
        r = (r_i + r_j) / 2

        # We now consider only the magnitude of the distances.
        R = la.norm(R)
        r = la.norm(r)

        # Calculate the sine-squared deformation.
        return (1 / 2) * (1 + np.cos(π * r / (R + 1 / 2)))

    return profile
