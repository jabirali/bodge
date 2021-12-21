import numpy as np
import functools as ft

from utils import *
from typing import Tuple

# from typing import
# from numpy.typing import ArrayLike

from numpy.linalg import eigh
from scipy.sparse import lil_matrix

σ = [
    *map(
        lambda s: np.array(s, dtype=np.complex64),
        ([[1, 0], [0, 1]], [[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]),
    )
]


class Hamiltonian(object):
    """Representation of the Hamiltonian matrix for a system.

    This system assumes that 
    """

    def __init__(self, size):
        # Allocate every member variable. These are consistently

        # System dimensions.
        self.dims: Tuple[int, int, int] = tuple(size)
        N = np.prod(size)

        # Physical fields.
        self.μ = np.zeros((N, 1))  # Chemical potential
        self.Δ = np.zeros((N, 1))  # Superconducting gap
        self.m = np.zeros((N, 3))  # Magnetic exchange field
        self.t = np.zeros((N, N))  # Hopping amplitudes

        # Hamiltonian.
        # self.H = lil_matrix((4*N, 4*N), dtype=np.complex64)
        self.H = np.zeros((4 * N, 4 * N), dtype=np.complex64)

    def asmatrix(self):
        """Construct a matrix representation."""
        N = self.H.shape[0] // 4

        # Handle the diagonal terms.
        for i in range(N):
            # Calculate the electronic contribution.
            self[(i, 0, 0), (i, 0, 0), 0, 0] = self.μ[i, 0]
            self[(i, 0, 0), (i, 0, 0), 1, 0] = self.m[i, 0]
            self[(i, 0, 0), (i, 0, 0), 2, 0] = self.m[i, 1]
            self[(i, 0, 0), (i, 0, 0), 3, 0] = self.m[i, 2]

            Δ_ii = self.Δ[i, 0] * (1j * σ[2])

            # Nambu⊗Spin matrix for these lattice sites.
            self.H[4 * i + 0 : 4 * i + 2, 4 * i + 0 : 4 * i + 2] = +H_ii
            self.H[4 * i + 2 : 4 * i + 4, 4 * i + 2 : 4 * i + 4] = -H_ii.conj()
            self.H[4 * i + 0 : 4 * i + 2, 4 * i + 2 : 4 * i + 4] = +Δ_ii
            self.H[4 * i + 2 : 4 * i + 4, 4 * i + 0 : 4 * i + 2] = +Δ_ii.T.conj()

        # TODO: Handle hopping.

        return self.H

    def __iter__(self):
        """Return an iterator over all coordinate pairs."""
        pass

    def __setitem__(self, args, val):
        """Human-readable setter for Hamiltonian matrix elements.

        This method defines the syntactic sugar required so that we can:
        - Specify positions via coords (x, y, z) instead of a flat index i;
        - Specify spinor structure via a Pauli decomposition (σ⁰, σ¹, σ², σ³)
          instead of explicit matrix structures like [[0, -1j], [+1j, 0]];
        - Specify one normal (electronic) and off-diagonal (superconducting)
          block of the Hamiltonian, and let the rest follow from symmetry.
        """
        # Extract two sets of coordinates from the arguments.
        (x_1, y_1, z_1), (x_2, y_2, z_2), spin, anomal = args

        # Convert the positional coordinates to flat indices.
        i = coord2index((x_1, y_1, z_1), self.dims)
        j = coord2index((x_2, y_2, z_2), self.dims)

        # Construct the appropriate matrix in spin space.
        H = val * pauli[spin]

        # Append the new matrix elements.
        if not anomal:
            # Normal contribution (electron-electron term).
            self.H[4*i + 0 : 4*i + 2, 4*j + 0 : 4*j + 2] += H
            self.H[4*i + 2 : 4*i + 4, 4*j + 2 : 4*j + 4] -= H.conj()
        else:
            # Anomalous contribution (superconductivity)
            self.H[4*i + 0 : 4*i + 2, 4*j + 2 : 4*j + 4] = +H
            self.H[4*i + 2 : 4*i + 4, 4*j + 0 : 4*j + 2] = +H.T.conj()

    # def __setattr__(self, key, val):
    #     """Setter for physical fields."""
    #     if key in self.__dict__:
    #         self.__dict__[key] = val
    #     else:
    #         if key == 'Δs':
    #             self[*key, 1] = val

    def __getitem__(self, args):
        """Human-readable getter for Hamiltonian matrix elements.

        This function defines the syntactic sugar so that we can:
        - Specify positions via coords (iˣ, iʸ, iᶻ) instead of a flat index i;
        - Specify spinor structure via a Pauli decomposition (σᵒ, σˣ, σʸ, σᶻ)
          instead of explicit matrix structures like [[0, -1j], [+1j, 0]];
        - Specify one diagonal (electron) and off-diagonal (superconducting)
          block of the Hamiltonian, and the others then follow from symmetry.
        This greatly simplifies the getting and setting 
        """
        pass
        # Extract two state vectors with the format u = (x, y, z, s),
        # then unpack them into a position index i 
        # u_i, u_j = args
        # i = coord2index(u[:-1], self.dims)
        # j = coord2index(v[:-1], self.dims)
        # s_i =

        # r_i, s_i = u_i[:-1], u_i[-1]
        # r_j, s_j = u_j[:-1], u_j[-1]

        # uⁱ, uʲ = args

        # rⁱ, sⁱ = uⁱ[:-1], uʲ[:-1]
        # rⁱ, rʲ = uⁱ[-1], uʲ[-1]
        # sⁱ = uⁱ[-1]

        # r_i = u_i[:-1]
        # s_i = 

        # # Calculate the corresponding flattened indices.
        # i = coord2index(r_i, self.dims)
        # j = coord2index(r_j, self.dims)
        # return((i, j))


G = Hamiltonian([2, 1, 1])

# G.H[(0, 0, 0), (0, 0, 0), 2].H = 1

# G[(0, 0, 0), (0, 0, 0), 2, 0] = 1
# G[(1, 0, 0), (1, 0, 0), 2, 0] = 2
# G[(0, 0, 0), (1, 0, 0), 0, 1] = 3
# G[(1, 0, 0), (0, 0, 0), 0, 1] = 4
print(G.H)


# G.μ[:, 0] = 1
# G.m[:, 1] = 2
# G.Δ[:, 0] = 3

# with np.printoptions(precision=0, suppress=True):
#     print(G.asmatrix())

#     E, X = eigh(G.asmatrix())
#     χ = [X[:,n] for n, E_n in enumerate(E) if E_n > 0]
#     E = [E_n for E_n in E if E_n > 0]
#     print(E)
#     print(χ)
