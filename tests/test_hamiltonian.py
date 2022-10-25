import numpy as np
from scipy.linalg import eigh

from bodge.hamiltonian import *
from bodge.lattice import *


def test_hermitian():
    # Instantiate a somewhat dense complex Hamiltonian. Note that
    # the submatrices need to be Hermitian for the whole to be so.
    lattice = CubicLattice((3, 5, 7))
    system = Hamiltonian(lattice)

    with system as (H, Δ, _):
        for i in lattice.sites():
            H[i, i] = 1 * σ3 + 2 * σ2
            Δ[i, i] = 5 * σ0 - 3 * σ2

        for i, j in lattice.bonds():
            H[i, j] = 3 * σ0 - 4 * σ2
            Δ[i, j] = 2 * σ3 + 5 * σ2

    # Verify that the result is Hermitian.
    H = system.matrix.todense()
    assert np.allclose(H, H.T.conj())


def test_sparsity():
    # Instantiate a somewhat random test system.
    lattice = CubicLattice((3, 5, 7))
    system = Hamiltonian(lattice)

    with system as (H, Δ, _):
        for i in lattice.sites():
            H[i, i] = 1 * σ3 + 2 * σ2
            Δ[i, i] = 5 * σ0 - 3 * σ2

        for i, j in lattice.bonds():
            H[i, j] = 3 * σ0 - 4 * σ2
            Δ[i, j] = 2 * σ3 + 5 * σ2

    # Calculate a matrix product using internal matrices.
    H = system.matrix
    S = system.struct
    I = system.identity
    G = H @ I

    # Ensure that the Hamiltonian H has a 4x4 BSR representation.
    assert H.blocksize == (4, 4)

    # Ensure that the identity preserves format and value.
    assert G.blocksize == (4, 4)
    assert np.allclose(G.todense(), H.todense())

    # Ensure the structure matrix is consistent with Hamiltonian.
    H.data[...] = 1
    assert S.blocksize == (4, 4)
    assert np.allclose(H.todense(), S.todense())
