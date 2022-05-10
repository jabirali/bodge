from .solver import *


class chebyshev(BlockSolver):
    """Chebyshev expansion of spectral functions."""

    def __init__(self, *args, **kwargs):
        # Superclass constructor.
        super().__init__(*args, **kwargs)

    def solve(self, block: int):
        """Chebyshev expansion of a given block of the spectral function."""
        # Chebyshev nodes {ω_m} where we calculate the spectral function
        # and corresponding weights {w_m} used for quadrature integration.
        N = 200
        k = np.arange(2 * N)
        ω = np.cos(π * (2 * k + 1) / (4 * N))
        w = N * π * np.sqrt(1 - ω**2)

        # Calculate the corresponding Chebyshev transform coefficients.
        # TODO: Incorporate the relevant Lorentz kernel factors here.
        n = np.arange(N)
        T = np.cos(n[None, :] * np.arccos(ω[:, None])) / N
        T[:, 1:] *= 2

        # Compact notation for the essential matrices.
        H = self.hamiltonian
        I_k = self.block_identity
        P_k = self.block_neighbors
        R_k = self.block_subspace

        # Initialize the first two Chebyshev matrix blocks A_kn, and calculate
        # the corresponding contributions to the spectral function A_k(ω_m).
        A_k0 = I_k
        A_k1 = H @ I_k

        # Prepare a storage file for this block, and store the initial results.
        A_k = {}
        for m, _ in enumerate(ω):
            A_k[m] = T[m, 0] * A_k0 + T[m, 1] * A_k1

        # Chebyshev expansion of the next elements.
        for n in range(2, N):
            # Chebyshev expansion of next vector. Element-wise multiplication
            # by R_k projects the result back into the Local Krylov subspace.
            A_k1, A_k0 = (H @ A_k1).multiply(R_k) - A_k0, A_k1

            # Perform the Chebyshev transformation. Element-wise multiplication
            # by H_k preserves only on-site and nearest-neighbor interactions.
            # WARNING: This has been optimized to ignore SciPy wrapper checks.
            AH_kn = A_k1.multiply(P_k)
            for m, A_km in A_k.items():
                A_k[m].data += T[m, n] * AH_kn.data

        # Calculate the integral from unweighted results.
        with File(self.blockname, "w") as file:
            B_k = sum(A_k.values())
            pack(file, f"/integral", B_k)

            if self.resolve:
                # Scale the final results using the integral weights.
                for m, A_km in A_k.items():
                    A_km.data /= w[m]

                # Save energy-resolved information.
                pack(file, f"/weights", w)
                pack(file, f"/energies", ω)
                for m, A_km in A_k.items():
                    pack(file, f"/spectral/{m:04d}", A_km)
