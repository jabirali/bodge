from bodge import *
from bodge.common import *
from pytest import raises


def test_hermitian():
    # Instantiate a somewhat dense complex Hamiltonian. Note that
    # the submatrices need to be Hermitian for the whole to be so.
    lattice = CubicLattice((3, 5, 7))
    system = Hamiltonian(lattice)

    with system as (H, Δ):
        for i in lattice.sites():
            H[i, i] = 1 * σ3 + 2 * σ2
            Δ[i, i] = 5 * σ0 - 3 * σ2

        for i, j in lattice.bonds():
            H[i, j] = 3 * σ0 - 4 * σ2
            Δ[i, j] = 2 * σ3 + 5 * σ2

        for i, j in lattice.edges():
            H[i, j] = 7 * σ0 - 13 * σ2
            Δ[i, j] = 9 * σ3 + 11 * σ2

    # Verify that the result is Hermitian.
    H = system.matrix.todense()
    assert np.allclose(H, H.T.conj())

    # Check non-hermiticity warnings.
    with raises(RuntimeError):
        i = (1, 1, 1)
        with system as (H, Δ):
            H[i, i] = 1j * σ1


def test_compilation():
    # Instantiate a somewhat random test system.
    lattice = CubicLattice((3, 5, 7))
    system = Hamiltonian(lattice)

    with system as (H, Δ):
        for i, j in lattice:
            H[i, j] = 3 * σ0 - 4 * σ2
            Δ[i, j] = 2 * σ3 + 5 * σ2

    # Construct matrix instances in different formats.
    H_BSR = system(format="bsr")
    H_CSR = system(format="csr")
    H_CSC = system(format="csc")
    H_DNS = system(format="dense")

    # Check that the dense matrix looks right.
    assert np.real(H_DNS[0, 0]) == 3
    assert np.imag(H_DNS[0, 1]) == 4
    assert np.real(H_DNS[0, 2]) == 2
    assert np.imag(H_DNS[0, 3]) == -5

    # Test that each construction is equivalent.
    assert np.max(np.abs(H_BSR - H_DNS)) < 1e-6
    assert np.max(np.abs(H_CSR - H_DNS)) < 1e-6
    assert np.max(np.abs(H_CSC - H_DNS)) < 1e-6

    # Check that the BSR blocksize is correct.
    assert H_BSR.blocksize == (4, 4)

    # Test error handling.
    with raises(Exception):
        H = system(format="blah")
    with raises(Exception):
        H = system(format=1)
