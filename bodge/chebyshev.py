from .solver import *


class chebyshev(Kernel):
    """Chebyshev expansion of spectral functions."""

    def solve(self) -> None:
        # The energies {ω_m} and integral weights {w_m} are determined from
        # the energy scale Ω and number N. The energies are not uniformly
        # distributed, but rather selected to coincide with the Chebyshev
        # nodes, which moreover simplifies later quadrature integration.
        Ω = self.scale
        M = self.energies
        m = np.arange(M)

        ω = Ω * np.cos(π * (m + 1 / 2) / M)
        w = π * np.sqrt(Ω**2 - ω**2)

        # Calculate the corresponding Chebyshev transform coefficients.
        # Since {ω_m} are Chebyshev nodes this is just a Fourier transform.
        N = M // 2
        n = np.arange(N)

        T = np.cos(π * n[None, :] * (m[:, None] + 1 / 2) / M)
        T[:, 1:] *= 2

        # Adjust the Chebyshev transform using a Jackson kernel. This avoids
        # Runge oscillations near sharp corners in the spectral functions.
        ϕ = π / (N + 1)
        g = (ϕ / π) * ((N - n + 1) * np.cos(ϕ * n) + np.sin(ϕ * n) / np.tan(ϕ))
        T *= g

        # Compact notation for the essential matrices.
        H = self.hamiltonian
        I_k = self.block_identity
        P_k = self.block_neighbors
        R_k = self.block_subspace * 2

        # Initialize the first two Chebyshev matrix blocks α_kn, and calculate
        # the corresponding contributions to the spectral function α_k(ω_m).
        α_k0 = I_k
        α_k1 = H @ I_k

        α_k = {}
        for m in range(M):
            α_k[m] = T[m, 0] * α_k0 + T[m, 1] * α_k1

        # Chebyshev expansion of the next elements.
        for n in range(2, N):
            # Chebyshev expansion of next vector. Element-wise multiplication
            # by R_k projects the result back into the Local Krylov subspace.
            # WARNING: A factor 2x has been absorbed by the R_k matrix here.
            α_k1, α_k0 = (H @ α_k1).multiply(R_k) - α_k0, α_k1

            # Perform the Chebyshev transformation. Element-wise multiplication
            # by P_k preserves only on-site and nearest-neighbor interactions.
            Pα_k1 = α_k1.multiply(P_k)
            for m in range(M):
                α_k[m] += T[m, n] * Pα_k1

        # Save results to file.
        with File(self.blockname, "w") as file:
            # Energy-integrated spectral function.
            A_k = sum(α_k.values())

            pack(file, f"/integral", A_k)

            # Energy-resolved spectral function.
            if self.resolve:
                pack(file, f"/weights", w)
                pack(file, f"/energies", ω)
                for m, α_km in α_k.items():
                    a_km = α_km / w[m]
                    pack(file, f"/spectral/{m:04d}", a_km)
