import os

import numpy as np
from h5py import File

from bodge import *
from bodge.solver import *


def populate(system):
    """Create a simple test Hamiltonian."""
    with system as (H, Δ):
        for i in system.lattice.sites():
            H[i, i] = 1 * σ3 + 2 * σ2
            Δ[i, i] = 5 * σ0 - 3 * σ2

        for i, j in system.lattice.bonds():
            H[i, j] = 3 * σ0 - 4 * σ2
            Δ[i, j] = 2 * σ3 + 5 * σ2


def test_blocking():
    """Test that division into matrix blocks for kernels works as intended."""
    # Instantiate a test system.
    lattice = CubicLattice((16, 8, 8))
    system = Hamiltonian(lattice)
    populate(system)

    # Determine blocking parameters.
    energies = 128
    blocksize = 128
    blocks = system.matrix.shape[0] // blocksize
    radius = 3

    # Save a test file to disk.
    filename = "test_blocking.hdf"
    with File(filename, "w") as file:
        pack(file, "/hamiltonian/matrix", system.matrix)
        pack(file, "/hamiltonian/struct", system.struct)
        pack(file, "/hamiltonian/scale", system.scale)
        pack(file, "/numerics/energies", energies)
        pack(file, "/numerics/blocksize", blocksize)
        pack(file, "/numerics/blocks", blocks)
        pack(file, "/numerics/radius", radius)
        pack(file, "/numerics/resolve", False)

    # Instantiate the kernel for a given block.
    kernel = Kernel(filename)
    kernel.solve = lambda: None

    # Test that each block of the identity matrix has the right dimensions.
    I = []
    H = system.matrix
    for k in range(blocks):
        kernel(k)
        I_k = kernel.block_identity

        assert I_k.blocksize == H.blocksize
        assert I_k.shape == (H.shape[0], kernel.blocksize)

        I.append(I_k)

    # Test that the reconstructed identity matrix looks correct.
    I = sp.hstack(I)
    assert np.allclose(I.todense(), system.identity.todense())

    # Check that the projection matrices are consistent with a manual expansion.
    for k in range(blocks):
        kernel(k)
        I_k = kernel.block_identity

        N_k = H @ I_k
        N_k.data[...] = 1
        assert np.allclose(N_k.todense(), kernel.block_neighbors.todense())

        S_k = H @ (H @ (H @ I_k))
        S_k.data[...] = 1
        assert np.allclose(S_k.todense(), kernel.block_subspace.todense())

    # Delete test files.
    os.remove(filename)
