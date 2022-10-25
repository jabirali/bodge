import numpy as np
from scipy.linalg import eigh

from bodge import *


def test_diagonalize():
    # Instantiate a system with superconductivity and a barrier.
    lattice = CubicLattice((10, 10, 1))
    system = Hamiltonian(lattice)

    with system as (H, Δ, _):
        for i in lattice.sites():
            H[i, i] = 4 * σ0
            if i[0] > 5:
                Δ[i, i] = 1 * jσ2
            elif i[0] > 3:
                H[i, i] += 6 * σ0
        for i, j in lattice.bonds():
            H[i, j] = -1 * σ0

    # Calculate the eigenvalues the manual way.
    H = system.scale * system.matrix.todense()
    E, X = eigh(H, subset_by_value=(0, np.inf))
    X = X.T

    # Confirm that we got positive eigenvalues and that we have
    # interpreted the corresponding eigenvector matrix correctly.
    assert E.size == 200
    for n, E_n in enumerate(E):
        assert E_n > 0
        assert np.allclose(H @ X[n, :], E_n * X[n, :])

    # Calculate the same eigenvalues via the package, and ensure
    # that the eigenvalues and eigenvectors are consistent.
    eigval, eigvec = diagonalize(system)
    assert np.allclose(eigval, E)
    for n, E_n in enumerate(E):
        for m in range(100):
            assert np.allclose(eigvec[n, m, 0], X[n, 4 * m + 0])
            assert np.allclose(eigvec[n, m, 1], X[n, 4 * m + 1])
            assert np.allclose(eigvec[n, m, 2], X[n, 4 * m + 2])
            assert np.allclose(eigvec[n, m, 3], X[n, 4 * m + 3])


def test_dvector_basic():
    """Brute-force test all d(p) = e_i p_j cases."""
    Δ = dvector("e_x * p_x")
    assert np.allclose(Δ((0, 0, 0), (1, 0, 0)), σ1 @ jσ2)
    assert np.allclose(Δ((0, 0, 0), (0, 1, 0)), 0)
    assert np.allclose(Δ((0, 0, 0), (0, 0, 1)), 0)

    Δ = dvector("e_x * p_y")
    assert np.allclose(Δ((0, 0, 0), (1, 0, 0)), 0)
    assert np.allclose(Δ((0, 0, 0), (0, 1, 0)), σ1 @ jσ2)
    assert np.allclose(Δ((0, 0, 0), (0, 0, 1)), 0)

    Δ = dvector("e_x * p_z")
    assert np.allclose(Δ((0, 0, 0), (1, 0, 0)), 0)
    assert np.allclose(Δ((0, 0, 0), (0, 1, 0)), 0)
    assert np.allclose(Δ((0, 0, 0), (0, 0, 1)), σ1 @ jσ2)

    Δ = dvector("e_y * p_x")
    assert np.allclose(Δ((0, 0, 0), (1, 0, 0)), σ2 @ jσ2)
    assert np.allclose(Δ((0, 0, 0), (0, 1, 0)), 0)
    assert np.allclose(Δ((0, 0, 0), (0, 0, 1)), 0)

    Δ = dvector("e_y * p_y")
    assert np.allclose(Δ((0, 0, 0), (1, 0, 0)), 0)
    assert np.allclose(Δ((0, 0, 0), (0, 1, 0)), σ2 @ jσ2)
    assert np.allclose(Δ((0, 0, 0), (0, 0, 1)), 0)

    Δ = dvector("e_y * p_z")
    assert np.allclose(Δ((0, 0, 0), (1, 0, 0)), 0)
    assert np.allclose(Δ((0, 0, 0), (0, 1, 0)), 0)
    assert np.allclose(Δ((0, 0, 0), (0, 0, 1)), σ2 @ jσ2)

    Δ = dvector("e_z * p_x")
    assert np.allclose(Δ((0, 0, 0), (1, 0, 0)), σ3 @ jσ2)
    assert np.allclose(Δ((0, 0, 0), (0, 1, 0)), 0)
    assert np.allclose(Δ((0, 0, 0), (0, 0, 1)), 0)

    Δ = dvector("e_z * p_y")
    assert np.allclose(Δ((0, 0, 0), (1, 0, 0)), 0)
    assert np.allclose(Δ((0, 0, 0), (0, 1, 0)), σ3 @ jσ2)
    assert np.allclose(Δ((0, 0, 0), (0, 0, 1)), 0)

    Δ = dvector("e_z * p_z")
    assert np.allclose(Δ((0, 0, 0), (1, 0, 0)), 0)
    assert np.allclose(Δ((0, 0, 0), (0, 1, 0)), 0)
    assert np.allclose(Δ((0, 0, 0), (0, 0, 1)), σ3 @ jσ2)


def test_dvector_pwave():
    """Check the p-wave property Δ(i, j) = -Δ(j, i)."""
    for desc in [
        "e_x * p_x",
        "e_z * p_y",
        "e_y * jp_z",
        "e_z * (p_x + jp_y)",
        "(e_x + je_y) * (p_y + jp_z)",
    ]:
        Δ = dvector(desc)
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


def test_dvector_hermitian():
    """Test that d-vector-constructed Hamiltonians are Hermitian."""
    lattice = CubicLattice((10, 10, 1))
    system = Hamiltonian(lattice)

    with system as (H, Δ, V):
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
        Δ_p = dvector(desc)
        with system as (H, Δ, V):
            for i, j in lattice.bonds():
                Δ[i, j] = -0.1 * Δ_p(i, j)

        H = system.matrix.todense()
        assert np.allclose(H, H.T.conj())