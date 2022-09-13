import numpy as np

from .hamiltonian import Hamiltonian
from .math import *
from .typing import *


class FermiMatrix:
    def __init__(self, hamiltonian: Hamiltonian, order: int):
        # Store initialization arguments.
        self.hamiltonian: Hamiltonian = hamiltonian
        self.order: int = order

        # Storage for the Fermi matrix.
        self.matrix: Optional[bsr_matrix] = None

    def __call__(self, temperature: float, radius: Optional[int] = None):
        # Hamiltonian and identity matrix.
        H = self.hamiltonian.matrix
        I = self.hamiltonian.identity

        # TODO: Generate identity blocks I_k instead of using full I
        # TODO: Perform (parallelizable) expansion of F_nk = [f_n T_n(X)]_k

        # Generators for coefficients and matrices.
        fs = fermi_coeff(temperature, self.order)
        gs = jackson_kernel(self.order)
        Ts = chebyshev(H, I, self.order, radius)

        # Initialize the Fermi matrix skeleton.
        self.matrix = bsr_matrix(H.shape, blocksize=H.blocksize, dtype=H.dtype)

        # Perform kernel polynomial expansion.
        for f, g, T in zip(fs, gs, Ts):
            if f != 0:
                self.matrix += f * g * T

        return self.matrix
