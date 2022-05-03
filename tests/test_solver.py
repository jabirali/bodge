import sys

import numpy as np

from bodge import *
from bodge.solver import *


def populate(system):
    with system as (H, Δ):
        for i in system.lattice.sites():
            H[i, i] = 1 * σ3 + 2 * σ2
            Δ[i, i] = 5 * σ0 - 3 * σ2

        for i, j in system.lattice.bonds():
            H[i, j] = 3 * σ0 - 4 * σ2
            Δ[i, j] = 2 * σ3 + 5 * σ2


def test_order():
    # Instantiate a test system.
    lattice = CubicLattice((8, 8, 16))
    system = Hamiltonian(lattice)
    solver1 = SpectralSolver(system)

    populate(system)

    solver2 = SpectralSolver(system)

    # Test that the Hamiltonians are consistent.
    assert np.allclose(solver1.hamiltonian.todense(), solver2.hamiltonian.todense())
    assert np.allclose(solver1.skeleton.todense(), solver2.skeleton.todense())

    # Generate an arbitrary block of the matrix.
    solver1.init_block(3)
    solver2.init_block(3)

    # Ensure that all block matrices are consistent.
    assert np.allclose(solver1.block_identity.todense(), solver2.block_identity.todense())
    assert np.allclose(solver1.block_neighbors.todense(), solver2.block_neighbors.todense())
    assert np.allclose(solver1.block_subspace.todense(), solver2.block_subspace.todense())


def test_blocking():
    # Instantiate a test system.
    lattice = CubicLattice((16, 8, 8))
    system = Hamiltonian(lattice)
    solver = SpectralSolver(system, radius=3)
    populate(system)

    H = system.matrix

    # Test that each block of the identity matrix has the right dimensions.
    I = []
    for k in range(solver.blocks):
        solver.init_block(k)
        I_k = solver.block_identity

        assert I_k.blocksize == H.blocksize
        assert I_k.shape == (H.shape[0], solver.blocksize)

        I.append(I_k)

    # Test that the reconstructed identity matrix looks correct.
    I = sp.hstack(I)
    assert np.allclose(I.todense(), system.identity.todense())

    # Check that the projection matrices are consistent with a manual expansion.
    for k in range(solver.blocks):
        solver.init_block(k)
        I_k = solver.block_identity

        N_k = H @ I_k
        N_k.data[...] = 1
        assert np.allclose(N_k.todense(), solver.block_neighbors.todense())

        S_k = H @ (H @ (H @ I_k))
        S_k.data[...] = 1
        assert np.allclose(S_k.todense(), solver.block_subspace.todense())

    # Manually calculate block matrices.
    # k = 3
    # H = system.matrix
    # I = system.
    # I

    # Test that the
    # solver = SpectralSolver(system)
