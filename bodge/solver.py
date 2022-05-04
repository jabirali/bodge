from multiprocessing import Pool, cpu_count

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp
from tqdm import trange

from .consts import *
from .lattice import *
from .physics import *


class SpectralSolver:
    """Defines an API for numerically calculating spectral functions.

    This solver assumes the following properties for the derived solvers:

    - The Hamiltonian and spectral function are implemented as BSR matrices;
    - Linear-scaling algorithms based on a Local Krylov subspace are employed;
    - Parallelization based on the independent expansion of matrix blocks.

    To use this class, subclass it and implement the missing solver method.
    """

    def __init__(
        self,
        hamiltonian: Hamiltonian,
        processes: Optional[int] = None,
        blocksize: int = 1024,
        radius: int = 4,
    ):
        # Reference to the sparse matrix we use.
        self.hamiltonian: sp.bsr_matrix = hamiltonian.matrix
        self.skeleton: sp.bsr_matrix = hamiltonian.struct

        # Linear scaling is achieved via a Local Krylov cutoff.
        self.radius: int = radius
        if self.radius < 1:
            raise RuntimeError("Krylov cutoff radius must be a positive integer.")

        # Parallelization is done by division into matrix blocks.
        self.blocksize: int = blocksize
        self.blocks: int = self.hamiltonian.shape[1] // blocksize
        if self.blocksize * self.blocks != hamiltonian.shape[1]:
            raise RuntimeError(f"Hamiltonian shape must be a multiple of {blocksize}.")

        # By default we use one less process than number of processors (min 2).
        # This ensures the remaining CPU core is free to orchestrate the rest.
        if processes is not None:
            self.processes: int = processes
        else:
            self.processes: int = max(cpu_count() - 1, 2)

        # Declare additional variables for methods and subclasses to define.
        self.energies: npt.NDArray[np.float64]
        self.solution: list[sp.bsr_matrix] = []

        self.block_identity: sp.bsr_matrix
        self.block_neighbors: sp.bsr_matrix
        self.block_subspace: sp.bsr_matrix

    def __call__(self, block: Optional[int] = None):
        """Calculate the spectral function as a function of energy.

        After instantiating this class as e.g. `solver`, run `solver()` in
        the `__main__` process to calculate the complete spectral function.
        The results are stored in the `.solution` variable of the instance.

        Calling the instance as `solver(k)` calculates block number k, and
        is meant to be run by individial worker processes in parallel. The
        solver should usually not be used in this manner in user scripts.
        """
        if block is None:
            # Delete any previously calculated solutions to save memory.
            del self.solution

            # Calculate each block A_km = [A_k(ω_m)] in parallel, then use `zip`
            # to transpose from block-first (A_km) to energy-first (A_mk) format.
            with Pool(self.processes) as pool:
                blocks = zip(*pool.map(self, trange(self.blocks)))

            # Merge the parallel blocks into complete matrices and store these.
            self.solution = [sp.hstack(block, "bsr") for block in blocks]
        else:
            # Calculate the spectral function A_k(ω_m) for a given block index k.
            self.block_init(block)
            return self.block_solve()

    def block_init(self, block: int) -> None:
        """Prepare for performing calculations at a given block index.

        This instantiates all the matrices required for a linear-scaling
        polynomial expansion of the spectral function at the given block.
        """
        # Instantiate the current block of the identity matrix.
        diag = np.repeat(np.int8(1), self.blocksize)
        offset = -block * self.blocksize
        shape = (self.hamiltonian.shape[0], self.blocksize)
        identity = sp.dia_matrix((diag, [offset]), shape, dtype=np.int8)

        self.block_identity = identity.tobsr(self.hamiltonian.blocksize)

        # Projection with this mask retains only local terms (up to nearest
        # neighbors), which are the relevant terms in the spectral function.
        self.block_neighbors = self.skeleton @ self.block_identity

        # Projection with this mask retains all terms within a "bubble" of
        # a given radius. This defines the Local Krylov subspace used for
        # intermediate calculations in the Green function expansions.
        mask = self.block_neighbors
        for _ in range(self.radius - 1):
            mask = self.skeleton @ mask
        mask.data[...] = 1

        self.block_subspace = sp.bsr_matrix(mask, dtype=np.int8)

    def block_solve(self):
        raise NotImplementedError


class ChebyshevSolver(SpectralSolver):
    """This class facilitates a Chebyshev expansion of spectral functions.

    Specifically, it calculates the scaled function `a(ω) = A(ω) / Nπ sqrt(1-ω²)`,
    where `A(ω) = [Gᴬ(ω) - Gᴿ(ω)] / 2πi` is the spectral function. This is defined
    such that summing up the `a(ω_k)` at the Chebyshev nodes `ω_k` is equivalent
    to integrating `A(ω)` over the same range. This is useful for calculating e.g.
    an order parameter or current, but when spectral quantities such as the
    density of states are desired, this requires a scaling by `Nπ sqrt(1-ω²)`.

    The `radius` determines the size of the Local Krylov subspace used for the
    expansion, `moments` sets the number of Chebyshev matrices to include in
    the expansion, and `system` provides a previously configured Hamiltonian.
    """

    def __init__(self, *args, moments: int = 200, **kwargs):
        # Superclass constructor.
        super().__init__(*args, **kwargs)

        # Chebyshev nodes {ω_m} where we will calculate the spectral function.
        N = moments
        k = np.arange(2 * N)
        ω = np.cos(π * (2 * k + 1) / (4 * N))

        # Calculate the corresponding Chebyshev transform coefficients.
        # TODO: Incorporate the relevant Lorentz kernel factors here.
        # TODO: Reintroduce the scaling prefactor for DOS calcs.
        # TODO: Calculate and save integral weights.
        n = np.arange(N)
        T = np.cos(n[None, :] * np.arccos(ω[:, None])) / N
        T[:, 1:] *= 2

        # Save relevant variables internally.
        self.moments = moments
        self.chebyshev = T
        self.energies = ω

    def block_solve(self):
        """Chebyshev expansion of a given block of the spectral function."""
        # Compact notation for the essential matrices.
        H = self.hamiltonian
        T = self.chebyshev

        I_k = self.block_identity
        P_k = self.block_neighbors
        R_k = self.block_subspace

        # Initialize the first two Chebyshev matrix blocks A_kn, and calculate
        # the corresponding contributions to the spectral function A_k(ω_m).
        A_k0 = I_k
        A_k1 = H @ I_k

        A_k = [T[m, 0] * A_k0 + T[m, 1] * A_k1 for m in range(2 * self.moments)]

        # Chebyshev expansion of the next elements.
        R_k *= 2
        for n in range(2, self.moments):
            # Chebyshev expansion of next vector. Element-wise multiplication
            # by R_k projects the result back into the Local Krylov subspace.
            A_k1, A_k0 = (H @ A_k1).multiply(R_k) - A_k0, A_k1

            # Perform the Chebyshev transformation. Element-wise multiplication
            # by H_k preserves only on-site and nearest-neighbor interactions.
            # WARNING: This has been optimized to ignore SciPy wrapper checks.
            AH_kn = A_k1.multiply(P_k)
            for m, A_km in enumerate(A_k):
                A_km.data += T[m, n] * AH_kn.data

        return A_k
