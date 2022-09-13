import numpy as np
from tqdm import tqdm

from .hamiltonian import *
from .math import *
from .stdio import *
from .typing import *


class FermiMatrix:
    def __init__(self, hamiltonian: Hamiltonian, order: int):
        # Store initialization arguments.
        self.hamiltonian: Hamiltonian = hamiltonian
        self.order: int = order

        # Storage for the Fermi matrix.
        self.matrix: Optional[bsr_matrix] = None

    def __call__(self, temperature: float, radius: Optional[int] = None):
        """Calculate the Fermi matrix at a given temperature."""
        log(self, "Performing Fermi-Chebyshev expansion")

        # Hamiltonian and related matrices.
        H = self.hamiltonian.matrix
        S = self.hamiltonian.struct
        I = self.hamiltonian.identity

        # Generators for coefficients and matrices.
        fs = self.coeff(temperature)
        gs = jackson(self.order)
        Ts = chebyshev(H, I, self.order, radius)

        # Initialize the Fermi matrix skeleton.
        self.matrix = bsr_matrix(H.shape, blocksize=H.blocksize, dtype=H.dtype)

        # Perform kernel polynomial expansion.
        # TODO: Check adjustments for entropy.
        for f, g, T in tqdm(zip(fs, gs, Ts), total=self.order):
            if f != 0:
                self.matrix += (f * g * T).multiply(S)

        return self.matrix

    def coeff(self, temperature: float):
        """Chebyshev coefficients of the Fermi function at a given temperature.

        We define the coefficients f_n such that f(X) = ∑ f_n T_n(X) for any X,
        where the sum goes over 0 ≤ n < N and T_n(X) is found by `chebyshev`.
        """
        # Short-hand notation for parameters.
        N = self.order
        T = temperature

        # Calculate the φ_k such that ε_k = cos(φ_k) are Chebyshev nodes.
        k = np.arange(N)
        φ = π * (k + 1 / 2) / (2 * N)

        # This expansion follows from f(ε) = [1 - tanh(ε/2T)] / 2.
        yield 1 / 2
        for n in range(1, N):
            match n % 2:
                case 0:
                    yield 0
                case 1:
                    yield -np.mean(np.tanh(np.cos(φ) / (2 * T)) * np.cos(n * φ))
