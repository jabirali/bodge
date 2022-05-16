from .solver import *


class chebyshev(Kernel):
    """Chebyshev expansion of spectral functions."""

    def solve(self) -> None:
        # Chebyshev nodes {ω_m} where we calculate the spectral function
        # and corresponding weights {w_m} used for quadrature integration.
        N = self.energies // 2
        k = np.arange(2 * N)
        ω = np.cos(π * (2 * k + 1) / (4 * N))
        w = π * np.sqrt(1 - ω**2) 

        # Calculate the corresponding Chebyshev transform coefficients.
        # TODO: Incorporate the relevant Lorentz kernel factors here.
        n = np.arange(N)[None, :]
        k = np.arange(2 * N)[:, None]
        T = np.cos(π * n * (2 * k + 1) / (4 * N))

        # Include the Lorentz kernel.
        λ = 1
        g = np.sinh(λ * (1 - n / N)) / np.sinh(λ)
        T *= g
        # T[:, 1:] *= 2

        # Compact notation for the essential matrices.
        H = self.hamiltonian
        I_k = self.block_identity
        P_k = self.block_neighbors
        R_k = self.block_subspace * 2

        # Initialize the first two Chebyshev matrix blocks A_kn, and calculate
        # the corresponding contributions to the spectral function A_k(ω_m).
        A_k0 = I_k
        A_k1 = H @ I_k

        # Prepare a storage file for this block, and store the initial results.
        A_k = {}
        for m, _ in enumerate(ω):
            A_k[m] = (T[m, 0] * A_k0 + T[m, 1] * A_k1) / w[m]

        # Chebyshev expansion of the next elements.
        for n in range(2, N):
            # Chebyshev expansion of next vector. Element-wise multiplication
            # by R_k projects the result back into the Local Krylov subspace.
            A_k1, A_k0 = 2 * (H @ A_k1) - A_k0, A_k1
            # A_k1, A_k0 = (H @ A_k1).multiply(R_k) - A_k0, A_k1

            # Perform the Chebyshev transformation. Element-wise multiplication
            # by H_k preserves only on-site and nearest-neighbor interactions.
            # WARNING: This has been optimized to ignore SciPy wrapper checks.
            # AH_kn = A_k1.multiply(P_k)
            for m, A_km in A_k.items():
                A_k[m] += 2 * T[m, n] * A_k1 / w[m]
                # A_k[m].data += T[m, n] * AH_kn.data

        # Calculate the integral from unweighted results.
        with File(self.blockname, "w") as file:
            B_k = sum(A_k.values())
            pack(file, f"/integral", B_k)

            if self.resolve:
                # Scale the final results using the integral weights.
                # for m, A_km in A_k.items():
                #     A_km.data /= w[m]

                # Save energy-resolved information.
                pack(file, f"/weights", w)
                pack(file, f"/energies", ω)
                for m, A_km in A_k.items():
                    pack(file, f"/spectral/{m:04d}", A_km)
