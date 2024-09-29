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
    However, you can use the `.matrix()` method to export the result to other
    common sparse matrix formats (e.g. CSR) or a dense matrix (NumPy array).
    You can also use the `.diagonalize()` method to get the eigenvalues and
    eigenvectors of the Hamiltonian, although this is usually inefficient.

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
        self._matrix: BsrMatrix = BsrMatrix(skeleton, dtype=np.complex128)
        self._matrix.data[...] = 0

        # Simplify direct access to the underlying data structure.
        self._data: Matrix = self._matrix.data

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
        self._hopp = {}
        self._pair = {}

        return self._hopp, self._pair

    @typecheck
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Implement a context manager interface for the class.

        This part of the implementation takes care of finalizing the Hamiltonian:

        - Transferring elements from the context manager dicts to the sparse matrix;
        - Ensuring that particle-hole and nearest-neighbor symmetries are respected;
        - Verifying that the constructed Hamiltonian is actually Hermitian.
        """
        # Process hopping terms: H[i, j].
        for (i, j), val in self._hopp.items():
            # Find this matrix block.
            k = self.index(i, j)

            # Update respecting electron-hole symmetry.
            self._data[k, 0:2, 0:2] = +val
            self._data[k, 2:4, 2:4] = -val.conj()

        # Process pairing terms: Δ[i, j].
        for (i, j), val in self._pair.items():
            # Find this matrix block.
            k1 = self.index(i, j)
            k2 = self.index(j, i)

            # Update respecting Hermitian symmetry.
            self._data[k1, 0:2, 2:4] = +val
            self._data[k2, 2:4, 0:2] = +val.T.conj()

        # Verify that the matrix is Hermitian.
        if np.max(np.abs(self._matrix - self._matrix.getH())) > 1e-6:
            raise RuntimeError("The constructed Hamiltonian is not Hermitian!")

        # Reset accessors.
        del self._hopp
        del self._pair

    @typecheck
    def matrix(self, format: str = "dense") -> SpMatrix | Matrix:
        """Return a specific matrix representation of the Hamiltonian.

        If the format is set to "dense", the result is returned as a NumPy
        array. If it is set to one of {"bsr", "csr", "csc"}, then the
        corresponding SciPy sparse matrix format is used instead.
        """
        # Transform the stored matrix into the requested matrix format. Trim
        # any remaining zero entries if a sparse matrix is requested.
        match format:
            case "bsr":
                # NOTE: Don't run eliminate_zeros on the original matrix, as that
                # would preclude adding new elements to the Hamiltonian later.
                H: BsrMatrix = self._matrix.copy()
                H.eliminate_zeros()
            case "csr":
                H: CsrMatrix = self._matrix.tocsr()
                H.eliminate_zeros()
            case "csc":
                H: CscMatrix = self._matrix.tocsc()
                H.eliminate_zeros()
            case "dense":
                H: Matrix = self._matrix.todense()
            case _:
                raise RuntimeError("Requested matrix format is not yet supported")

        return H

    @typecheck
    def index(self, row: Coord, col: Coord) -> Index:
        """Determine the sparse matrix index corresponding to block (row, col).

        This can be used to access `self.data[index, :, :]` when direct
        changes to the encapsulated block-sparse matrix are required.
        """
        indices, indptr = self._matrix.indices, self._matrix.indptr

        i, j = self.lattice[row], self.lattice[col]
        js = indices[indptr[i] : indptr[i + 1]]
        k = indptr[i] + np.where(js == j)

        return Index(k[0, 0])

    @typecheck
    def diagonalize(
        self, cuda=False, format="classic"
    ) -> tuple[Matrix, Matrix] | dict[float, tuple[Matrix, Matrix, Matrix, Matrix]]:
        """Calculate the exact eigenstates of the system via direct diagonalization.

        If you run it as `E, v = system.diagonalize()`, then the eigenvalue `E[n]`
        corresponds to the eigenvector `v[n, :, :]`, where the remaining indices
        of the vector correspond to a position index and a Nambu⊗Spin index. The
        last one means that at a given lattice site `i`, `v[n, i, 0:3]` will give
        you the eigenvector components corresponding to indices {e↑, e↓, h↑, h↓}.

        This default behavior is referred to as `format="classic"`, and is
        obtained by reshaping the 4N-element long vectors that actually satisfy
        the eigenvalue equation `H @ v[n] == E[n] * v[n]`. To obtain the "actual"
        eigenvectors instead, pass the argument `format="raw"` to this method.

        There is one more option available, namely `format="wave"`. This results
        in the return value being a dictionary. This is structured such that if
        you run `eig = system.diagonalize(format="dict")`, then you can use
        `for E, (e_up, e_dn, h_up, h_dn) in eig.items(): ...` to iterate through
        the eigenvalues and eigenvectors in the system. For each eigenvalue `E`,
        we then return wave functions `(e_up, e_dn, h_up, h_dn)` that correspond
        to respectively {e↑, e↓, h↑, h↓} states. These have been reshaped such
        that `e_up[x, y, z]` corresponds to the spin-up electron wave function
        at coordinates `(x, y, z)` on the lattice, and so forth. This variant
        may be a bit slower than the other formats, but is likely the easiest
        to use as a base for calculating various physical observables in the
        system due to the close similarity to the analytical BdG equations.

        Please note that this function uses *dense* matrices and thus requires a
        lot of compute power and memory for large matrices. Many computations
        can be performed using sparse matrix algorithms instead, so if you have
        a lattice with millions of sites I would consider alternative methods.

        Alternatively, if you have an NVIDIA GPU available and install the
        optional dependency `cupy`, then you can set `cuda=True` when running
        this function to enable a very fast GPU-accelerated diagonalization.
        Keep in mind that this only works if the Hamiltonian matrix and all its
        eigenvectors fit into your available video memory.
        """
        # Convert to a dense matrix.
        H = self.matrix(format="dense")

        # Calculate eigenvalues and eigenvectors.
        if cuda:
            # GPU-accelerated branch using CuPy.
            try:
                # Import libraries.
                import cupy as cp
                import cupy.linalg as cla

                # Diagonalize using CUDA.
                H = cp.asarray(H)
                eigval, eigvec = cla.eigh(H)
                eigval = cp.asnumpy(eigval)
                eigvec = cp.asnumpy(eigvec)

                # Extract positive eigenvalues.
                ind = np.where(eigval > 0)
                eigval, eigvec = eigval[ind], eigvec[:, ind]
            except ModuleNotFoundError:
                raise RuntimeError(
                    "Optional dependency `cupy` must be installed to use the flag `cuda=True`."
                )
        else:
            # CPU fallback branch using SciPy.
            eigval, eigvec = la.eigh(
                H, subset_by_value=(0.0, np.inf), overwrite_a=True, driver="evr"
            )
            eigval = np.array(eigval)
            eigvec = np.array(eigvec)

        # Maybe return the raw eigenvalues and eigenvectors as-is.
        if format == "raw":
            return eigval, eigvec

        # Restructure the eigenvectors to have the format eigvec[n, i, α],
        # where n corresponds to eigenvalue E[n], i is a position index, and
        # α represents the combined particle and spin index {e↑, e↓, h↑, h↓}.
        eigvec = eigvec.T.reshape((eigval.size, -1, 4))

        # Maybe return the eigenvalues and reshaped eigenvectors.
        if format == "classic":
            return eigval, eigvec

        # Split the eigenvectors into 4 vectors.
        e_up = eigvec[:, :, 0]
        e_dn = eigvec[:, :, 1]
        h_up = eigvec[:, :, 2]
        h_dn = eigvec[:, :, 3]

        # Reshape the eigenvectors to fit the lattice.
        # NOTE: THIS PART NEEDS SOME SERIOUS TESTING!
        N = len(eigval)
        e_up = e_up.reshape((N, *self.lattice.shape))
        e_dn = e_dn.reshape((N, *self.lattice.shape))
        h_up = h_up.reshape((N, *self.lattice.shape))
        h_dn = h_dn.reshape((N, *self.lattice.shape))

        # Construct a dict that maps energies to wave functions.
        eig = {
            E_n: (e_up[n, ...], e_dn[n, ...], h_up[n, ...], h_dn[n, ...])
            for n, E_n in enumerate(eigval)
        }

        # Maybe return the eigenstates as such a dict.
        if format == "wave":
            return eig

        # If we ever get here, the user didn't specify a valid format...
        raise RuntimeError(f"Eigenstate format '{format}' is not yet supported.")

    @typecheck
    def free_energy(self, temperature: float = 0.0) -> float:
        """Calculate the Landau free energy for a given Hamiltonian.

        This is done by computing all the positive eigenvalues ε_n of the matrix,
        and subsequently evaluating the entropy contributions to the free energy.
        The resulting free energy is then formulated as F = U - TS, where U is
        the internal energy (calculated from ɛ_n), T is the provided temperature,
        and S is the system's entropy (calculated from ε_n and T).

        In general, U has a constant contribution as well which corresponds to
        the non-matrix parts to the Hamiltonian operator. This contribution
        can't be calculated by this method as it is not part of the Hamiltonian
        matrix. For non-selfconsistent calculations, you usually don't need this
        constant. However, since this "constant" contribution is generally a
        function of the mean fields, you need to add it to the return value of
        this function for selfconsistent calculations to be correct.

        The algorithm implemented here is explained in Appendix C of:

            Ouassou et al. PRB 109, 174506 (2024).
            DOI: 10.1103/PhysRevB.109.174506
        """
        T = temperature
        H = self.matrix(format="dense")

        # Calculate the eigenvalues via a dense parallel algorithm. My benchmarks
        # have shown that this is usually faster than using the sparse solver.
        ε = la.eigh(H, overwrite_a=True, eigvals_only=True, driver="evr")

        # Extract the positive eigenvalues.
        ε = ε[ε > 0]

        # Internal energy.
        U = -(1 / 2) * np.sum(ε)

        # Entropy contribution.
        if T == 0:
            S = 0
        elif T > 0:
            S = np.sum(np.log(1 + np.exp(-ε / T)))
        else:
            raise ValueError("Expected non-negative temperature!")

        # Free energy
        F = U - T * S

        return F

    @typecheck
    def ldos(self, site: Coord, energies: Matrix | list[float]) -> Matrix:
        """Calculate the local density of states via a resolvent operator approach.

        We define the resolvent operator as [(ε+iη)I - H] R(ε) = I, which can be
        divided into vectors R(ε) = [r_1 ... r_N] and I = [e_1 ... e_N]. Diagonal
        elements of the resolvent matrix correspond to the density of states. By
        calculating one vector at a time via a sparse linear solver `spsolve`, the
        local density of states can be efficiently calculated at specific points.

        Note that if the requested energies are symmetric around ε = 0, then only
        half the calculations need to be performed due to particle-hole symmetry.

        The algorithm implemented here is explained in detail in Appendix A of:

        Ouassou et al. PRB 109, 174506 (2024).
        DOI: 10.1103/PhysRevB.109.174506
        """
        # Sparse Hamiltonian and identity matrices.
        H = self.matrix(format="csc")
        I = sp.identity(H.shape[0], format="csc")

        # Ensure that the input is NumPy floats.
        energies = np.array(energies, dtype=float)

        # Determine which energies we need calculations for.
        ε = np.unique(np.abs(energies))

        # Determine broadening parameter from the provided energies.
        Γ = np.gradient(ε)

        # Construct a reduced identity matrix for the relevant coordinate.
        N = H.shape[1]
        M = 4

        i = self.lattice[site]
        rows = np.array([4 * i + α for α in range(4)])
        cols = np.arange(M)
        data = np.repeat(1, M)

        B = CscMatrix((data, (rows, cols)), shape=(N, M))

        # Calculate the density of states.
        ρ = {}
        for ε_n, Γ_n in zip(ε, Γ):
            # Solve the linear equations for the resolvent.
            A = (ε_n + 1j * Γ_n) * I - H
            X = spla.spsolve(A, B)

            # Extract the few elements of interest.
            x = X.multiply(B).sum(axis=0)

            # Calculate and store the density of states.
            e_up = x[0, 0]
            e_dn = x[0, 1]
            h_up = x[0, 2]
            h_dn = x[0, 3]

            ρ[+ε_n] = -np.imag(e_up + e_dn) / π
            ρ[-ε_n] = -np.imag(h_up + h_dn) / π

        # Extract only the requested values.
        ρ = np.array([ρ[ε_n] for ε_n in energies])

        return ρ


@typecheck
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


@typecheck
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


@typecheck
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


@typecheck
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

    TODO: Consider integrating this into `Hamiltonian.__exit__` to
    make it more transparent to the user (if it proves very useful).
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
