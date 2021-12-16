import numpy as np
import functools as ft

from scipy.sparse import lil_matrix

σ = [*map(lambda s: np.array(s, dtype=np.complex64),
    (
        [[1, 0], [0, 1]],
        [[0, 1], [1, 0]],
        [[0, -1j], [1j, 0]],
        [[1, 0], [0, -1]]
    ))]

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
        # self.H = lil_matrix((4*N, 4*N), dtype=np.complex64)
        self.H = np.zeros((4*N, 4*N), dtype=np.complex64)

    def asmatrix(self):
        """Construct a matrix representation."""
        N = self.H.shape[0] // 4

        # Handle the diagonal terms.
        for i in range(N):
            # Calculate the electronic contribution.
            H_ii = -(
                self.μ[i, 0] * σ[0] +
                self.m[i, 0] * σ[1] +
                self.m[i, 1] * σ[2] +
                self.m[i, 2] * σ[3]
            )

            Δ_ii = self.Δ[i, 0] * (1j*σ[2])

            # Nambu⊗Spin matrix for these lattice sites.
            self.H[4*i+0:4*i+2, 4*i+0:4*i+2] = +H_ii
            self.H[4*i+2:4*i+4, 4*i+2:4*i+4] = -H_ii.conj()
            self.H[4*i+0:4*i+2, 4*i+2:4*i+4] = +Δ_ii
            self.H[4*i+2:4*i+4, 4*i+0:4*i+2] = +Δ_ii.T.conj()

        # TODO: Handle hopping.

        return self.H

G = Hamiltonian([1,2])

G.μ[:, 0] = 1
G.m[:, 1] = 2
G.Δ[:, 0] = 3

with np.printoptions(precision=0, suppress=True):
    print(G.asmatrix())
