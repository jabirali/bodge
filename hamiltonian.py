import numpy as np
import functools as ft

class Hamiltonian(object):
    """TODO"""
    def __init__(self, size):
        # Allocate every member variable. These are consistently 

        # System dimensions.
        N = np.prod(size)

        # Physical fields.
        self.μ = np.zeros((N, 1))  # Chemical potential
        self.Δ = np.zeros((N, 1))  # Superconducting gap
        self.m = np.zeros((N, 3))  # Magnetic exchange field
        self.t = np.zeros((N, N))  # Hopping amplitudes

        # Hamiltonian.
        self.H = np.zeros((4*N, 4*N), dtype=np.float32)

    def asmatrix(self):
        """Construct a matrix representation."""
        return self.H

H = Hamiltonian([2,3])
print(H.asmatrix())
