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

def test_pwave_basic():
    """Brute-force test all d(p) = e_i p_j cases."""
    Δ = pwave("e_x * p_x")
    assert np.allclose(Δ((0, 0, 0), (1, 0, 0)), σ1 @ jσ2 / 2)
    assert np.allclose(Δ((0, 0, 0), (0, 1, 0)), 0)
    assert np.allclose(Δ((0, 0, 0), (0, 0, 1)), 0)

    Δ = pwave("e_x * p_y")
    assert np.allclose(Δ((0, 0, 0), (1, 0, 0)), 0)
    assert np.allclose(Δ((0, 0, 0), (0, 1, 0)), σ1 @ jσ2 / 2)
    assert np.allclose(Δ((0, 0, 0), (0, 0, 1)), 0)

    Δ = pwave("e_x * p_z")
    assert np.allclose(Δ((0, 0, 0), (1, 0, 0)), 0)
    assert np.allclose(Δ((0, 0, 0), (0, 1, 0)), 0)
    assert np.allclose(Δ((0, 0, 0), (0, 0, 1)), σ1 @ jσ2 / 2)

    Δ = pwave("e_y * p_x")
    assert np.allclose(Δ((0, 0, 0), (1, 0, 0)), σ2 @ jσ2 / 2)
    assert np.allclose(Δ((0, 0, 0), (0, 1, 0)), 0)
    assert np.allclose(Δ((0, 0, 0), (0, 0, 1)), 0)

    Δ = pwave("e_y * p_y")
    assert np.allclose(Δ((0, 0, 0), (1, 0, 0)), 0)
    assert np.allclose(Δ((0, 0, 0), (0, 1, 0)), σ2 @ jσ2 / 2)
    assert np.allclose(Δ((0, 0, 0), (0, 0, 1)), 0)

    Δ = pwave("e_y * p_z")
    assert np.allclose(Δ((0, 0, 0), (1, 0, 0)), 0)
    assert np.allclose(Δ((0, 0, 0), (0, 1, 0)), 0)
    assert np.allclose(Δ((0, 0, 0), (0, 0, 1)), σ2 @ jσ2 / 2)

    Δ = pwave("e_z * p_x")
    assert np.allclose(Δ((0, 0, 0), (1, 0, 0)), σ3 @ jσ2 / 2)
    assert np.allclose(Δ((0, 0, 0), (0, 1, 0)), 0)
    assert np.allclose(Δ((0, 0, 0), (0, 0, 1)), 0)

    Δ = pwave("e_z * p_y")
    assert np.allclose(Δ((0, 0, 0), (1, 0, 0)), 0)
    assert np.allclose(Δ((0, 0, 0), (0, 1, 0)), σ3 @ jσ2 / 2)
    assert np.allclose(Δ((0, 0, 0), (0, 0, 1)), 0)

    Δ = pwave("e_z * p_z")
    assert np.allclose(Δ((0, 0, 0), (1, 0, 0)), 0)
    assert np.allclose(Δ((0, 0, 0), (0, 1, 0)), 0)
    assert np.allclose(Δ((0, 0, 0), (0, 0, 1)), σ3 @ jσ2 / 2)


def test_pwave_symmetries():
    """Check the p-wave property Δ(i, j) = -Δ(j, i)."""
    for desc in [
        "e_x * p_x",
        "e_z * p_y",
        "e_y * jp_z",
        "e_z * (p_x + jp_y)",
        "(e_x + je_y) * (p_y + jp_z)",
    ]:
        Δ = pwave(desc)
        for x in range(3):
            for y in range(3):
                for z in range(3):
                    i0 = (x, y, z)
                    j1 = (x + 1, y, z)
                    j2 = (x, y + 1, z)
                    j3 = (x, y, z + 1)

                    assert np.allclose(+Δ(i0, j1), -Δ(j1, i0))
                    assert np.allclose(+Δ(i0, j2), -Δ(j2, i0))
                    assert np.allclose(+Δ(i0, j3), -Δ(j3, i0))


def test_pwave_hermitian():
    """Test that d-vector-constructed Hamiltonians are Hermitian."""
    lattice = CubicLattice((10, 10, 1))
    system = Hamiltonian(lattice)

    with system as (H, Δ):
        for i in lattice.sites():
            H[i, i] = -0.1 * σ0

        for i, j in lattice.bonds():
            H[i, j] = -1 * σ0

    for desc in [
        "e_x * p_x",
        "e_z * p_y",
        "e_y * jp_z",
        "e_z * (p_x + jp_y)",
        "(e_x + je_y) * (p_y + jp_z)",
    ]:
        Δ_p = pwave(desc)
        with system as (H, Δ):
            for i, j in lattice.bonds():
                Δ[i, j] = -0.1 * Δ_p(i, j)

        H = system.matrix.todense()
        assert np.allclose(H, H.T.conj())


def test_dwave_symmetries():
    """Test that the d_{x^2 - y^2} order parameter behaves as expected."""
    Δ_d = dwave()

    # Zero on-site contributions.
    assert np.allclose(Δ_d((0, 0, 0), (0, 0, 0)), 0 * σ0)
    assert np.allclose(Δ_d((1, 2, 3), (1, 2, 3)), 0 * σ0)

    # Zero z-axis contributions.
    assert np.allclose(Δ_d((0, 0, 0), (0, 0, 1)), 0 * σ0)
    assert np.allclose(Δ_d((0, 0, 1), (0, 0, 0)), 0 * σ0)

    # Positive x-axis contributions.
    assert np.allclose(Δ_d((0, 0, 0), (1, 0, 0)), +1 * jσ2)
    assert np.allclose(Δ_d((0, 0, 0), (9, 0, 0)), +1 * jσ2)
    assert np.allclose(Δ_d((1, 0, 0), (0, 0, 0)), +1 * jσ2)
    assert np.allclose(Δ_d((9, 0, 0), (0, 0, 0)), +1 * jσ2)

    # Negative y-axis contributions.
    assert np.allclose(Δ_d((0, 0, 0), (0, 1, 0)), -1 * jσ2)
    assert np.allclose(Δ_d((0, 0, 0), (0, 9, 0)), -1 * jσ2)
    assert np.allclose(Δ_d((0, 1, 0), (0, 0, 0)), -1 * jσ2)
    assert np.allclose(Δ_d((0, 9, 0), (0, 0, 0)), -1 * jσ2)

    # Zero diagonal contributions.
    assert np.allclose(Δ_d((1, +1, 0), (0, 0, 0)), 0 * σ0)
    assert np.allclose(Δ_d((1, -1, 0), (0, 0, 0)), 0 * σ0)
    assert np.allclose(Δ_d((0, 0, 0), (1, +1, 0)), 0 * σ0)
    assert np.allclose(Δ_d((0, 0, 0), (1, -1, 0)), 0 * σ0)


def test_dwave_hermitian():
    """Test that d-wave superconductors have Hermitian Hamiltonians."""
    lattice = CubicLattice((10, 10, 1))
    system = Hamiltonian(lattice)
    Δ_d = dwave()

    with system as (H, Δ):
        for i in lattice.sites():
            H[i, i] = -0.1 * σ0

        for i, j in lattice.bonds():
            H[i, j] = -1 * σ0
            Δ[i, j] = -0.1 * Δ_d(i, j)

    H = system.matrix.todense()
    assert np.allclose(H, H.T.conj())

def test_ssd():
    """Test the mathematical properties of the SSD profile."""
    lattice = CubicLattice((31, 137, 1))
    system = Hamiltonian(lattice)
    φ = ssd(system)

    # The profile goes to zero at corners.
    i = (0, 0, 0)
    assert np.allclose(φ(i, i), 0, atol=0.001)

    # The profile goes to one at origin.
    i = (15, 68, 0)
    assert np.allclose(φ(i, i), 1, atol=0.001)

    # Test the symmetry of the solution.
    # Useful to catch off-by-one errors.
    i = (0, 0, 0)
    j = (30, 136, 0)
    assert φ(i, i) == φ(j, j)

    # Test that coordinate averaging works.
    # The result here should be exact.
    i = (1, 21, 0)
    j = (11, 1, 0)
    k = (6, 11, 0)
    assert φ(i, j) == φ(k, k)
