from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
import scipy.sparse as sp
from tqdm import trange

from .consts import *
from .lattice import *
from .physics import *


class SpectralSolver:
    """Defines an API for numerically calculating spectral functions.

    This solver assumes the following properties for the derived solvers:
    - The Hamiltonian and spectral function are implemented as 4x4 BSR matrices.
    - The implementation is done using a Local Krylov cutoff for linear scaling.
    """

    def __init__(self, hamiltonian: Hamiltonian, blocksize: int = 1024, radius: int = 4):
        # Reference to the sparse matrix we use.
        self.hamiltonian: sp.bsr_matrix = hamiltonian.matrix

        # Linear scaling is achieved via a Local Krylov cutoff.
        self.radius: int = radius
        if self.radius < 1:
            raise RuntimeError("Krylov cutoff radius must be a positive integer.")

        # Parallelization is done by division into matrix blocks.
        self.blocksize: int = blocksize
        self.blocks: int = self.hamiltonian.shape[1] // blocksize
        if self.blocksize * self.blocks != hamiltonian.shape[1]:
            raise RuntimeError(f"Hamiltonian shape must be a multiple of {blocksize}.")

    def __call__(self, block: Optional[int] = None):
        if block is None:
            # Perform parallel calculations for all blocks.
            return self.calc_all()
        else:
            # Perform calculations for the current block.
            self.init_block(block)
            return self.calc_block(block)

    def init_block(self, block: int):
        # Construct the current block of the identity matrix.
        diag = np.repeat(np.int8(1), self.blocksize)
        offset = -block * self.blocksize
        shape = (self.hamiltonian.shape[0], self.blocksize)
        identity = sp.dia_matrix((diag, [offset]), shape, dtype=np.int8)

        self.identity: sp.bsr_matrix = identity.tobsr((4, 4))

        # Construct a projection mask containing nearest-neighbor interactions.
        mask_1 = self.hamiltonian @ self.identity
        mask_1.data[...] = 1

        self.mask_1: sp.spmatrix = sp.bsr_matrix(mask_1, dtype=np.int8)

        # Prepare matrices for the subspace expansion.
        # self.diag =

        # Prepare a cheap surrogate for the Hamiltonian.
        # This is primarily used to prepare matrix masks.

        # Generate the k'th slice of the projection mask.
        mask_r = self.mask_1
        for _ in range(2, self.radius):
            mask_r = self.hamiltonian @ mask_r
        mask_r.data[...] = 1

        self.mask_r: sp.spmatrix = sp.bsr_matrix(mask_1, dtype=np.int8)

        # self.identity = identity
        self.I_k = self.identity
        self.P_k = self.mask_r
        self.H_k = self.mask_1

    def calc_block(self, block: int):
        raise NotImplementedError

    def calc_all(self):
        raise NotImplementedError


class Chebyshev(SpectralSolver):
    """This class facilitates a Chebyshev expansion of Green functions.

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
        n = np.arange(N)
        T = np.cos(n[None, :] * np.arccos(ω[:, None])) / N
        T[:, 1:] *= 2

        # Save relevant variables internally.
        self.moments = moments

        self.transform = T
        self.energies = ω

    def calc_all(self):
        # Determine number of processes to use.
        jobs = cpu_count()

        # Calculate the spectral function in parallel.
        A = {}
        with Pool(jobs) as p:
            for k, A_k in p.imap(self, trange(self.blocks)):
                A[k] = A_k

        # Transpose and merge the calculated matrices.
        A = {
            ω_m: sp.hstack([A_k[m] for _, A_k in A.items()], "bsr")
            for m, ω_m in enumerate(self.energies)
        }

        # TODO: Calculate and return integral weights.

        return A

    def calc_block(self, block):
        """Chebyshev expansion of a given Green function block."""
        # Compact notation for relevant variables.
        k = block
        # M = self.H0
        I_k = self.identity
        P_k = self.P_k
        H_k = self.H_k

        # # Generate a slice of the identity matrix.
        # I_k = sp.dia_matrix(
        #     (self.diag, [-block * self.blocksize]),
        #     (self.hamiltonian.shape[0], self.blocksize),
        # ).tobsr((4, 4))

        # # Generate the k'th slice of the Hamiltonian mask.
        # H_k = M @ I_k
        # H_k.data[...] = 1

        # # Generate the k'th slice of the projection mask.
        # P_k = H_k
        # for _ in range(2, self.radius):
        #     P_k = M @ P_k
        # P_k.data[...] = 1

        # Shorter names for stored stuff.
        H = self.hamiltonian
        N = self.moments
        T = self.transform

        # Initialize the first two Chebyshev matrices needed to start recursion.
        G_k0 = I_k
        G_kn = H @ I_k

        # Green function slices G_k(ω_m) at the Chebyshev nodes ω_m. These are
        # initialized using the first two Chebyshev matrices defined above. No
        # projection is needed here since H_k and G_kn have the same structure.
        G_k = [T[m, 0] * G_k0 + T[m, 1] * G_kn for m in range(2 * N)]

        # Multiply the projection operator by 2x to fit the recursion relation.
        P_k *= 2

        # Chebyshev expansion of the next elements.
        for n in range(2, N):
            # Chebyshev expansion of next vector. Element-wise multiplication
            # by P_k projects the result back into the Local Krylov subspace.
            G_kn, G_k0 = (H @ G_kn).multiply(P_k) - G_k0, G_kn

            # Perform the Chebyshev transformation. Element-wise multiplication
            # by H_k preserves only on-site and nearest-neighbor interactions.
            # WARNING: This has been optimized to ignore SciPy wrapper checks.
            GH_kn = G_kn.multiply(H_k)
            for m, G_km in enumerate(G_k):
                G_km.data += T[m, n] * GH_kn.data

        return k, G_k
