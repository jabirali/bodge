"""Unit tests for the `Hamiltonian` class and some of its core methods.

In this set of tests, we mainly try to verify that the implementation
is mathematically correct. For more physically minded tests, please
see the integration tests in `test_physics.py`.
"""

import numpy as np
from numpy.random import random as r
from pytest import importorskip, raises
from scipy.linalg import eigh

from bodge import *
from bodge.common import *


def test_matrix_hermitian():
    """Test that a semi-random Hamiltonian matrix becomes Hermitian.

    Note that for the constructed Hamiltonian to be Hermitian, we
    need (i) a symmetry between hopping i->j and j->i, and (ii)
    each submatrix we define below to itself be Hermitian. Within
    these bounds, we add some randomness for more general testing.

    For non-Hermitian matrices, we test that an exception is raised.
    """
    # Instantiate a somewhat dense complex Hamiltonian. Note that
    # the submatrices need to be Hermitian for the whole to be so.
    lattice = CubicLattice((3, 5, 7))
    system = Hamiltonian(lattice)

    with system as (H, Δ):
        for i in lattice.sites():
            H[i, i] = r() * σ0 + r() * σ1 + r() * σ2 + r() * σ3
            Δ[i, i] = (r() * σ1 + r() * σ2 + r() * σ3) @ jσ2

        for i, j in lattice.bonds():
            t = [r(), r(), r(), r()]
            H[i, j] = t[0] * σ0 + t[1] * σ1 + t[2] * σ2 + t[3] * σ3
            H[j, i] = t[0] * σ0 + t[1] * σ1 + t[2] * σ2 + t[3] * σ3
            Δ[i, j] = (r() * σ1 + r() * σ2 + r() * σ3) @ jσ2

        for i, j in lattice.edges():
            T = [r(), r(), r(), r()]
            H[i, j] = T[0] * σ0 + T[1] * σ1 + T[2] * σ2 + T[3] * σ3
            H[j, i] = T[0] * σ0 + T[1] * σ1 + T[2] * σ2 + T[3] * σ3
            Δ[i, j] = (r() * σ1 + r() * σ2 + r() * σ3) @ jσ2

    # Verify that the result is Hermitian.
    H = system._matrix.todense()
    assert np.allclose(H, H.T.conj())

    # Check non-hermiticity warnings.
    with raises(Exception):
        i = (1, 1, 1)
        with system as (H, Δ):
            H[i, i] = 1j * σ1


def test_matrix_export():
    """Test that the Hamiltonian matrix is correctly exported.

    This entails that the same matrix should be constructed both when
    exporting to sparse and dense formats, and that an exception
    should be raised if an unimplemented format is requested.
    """

    # Instantiate a somewhat random test system.
    lattice = CubicLattice((3, 5, 7))
    system = Hamiltonian(lattice)

    with system as (H, Δ):
        for i, j in lattice:
            H[i, j] = 3 * σ0 - 4 * σ2
            Δ[i, j] = 2 * σ3 + 5 * σ2

    # Construct matrix instances in different formats.
    H_DNS = system.matrix(format="dense")
    H_BSR = system.matrix(format="bsr")
    H_CSR = system.matrix(format="csr")
    H_CSC = system.matrix(format="csc")

    # Check that each matrix has the right format.
    assert isinstance(H_DNS, np.ndarray)
    assert H_BSR.getformat() == "bsr"
    assert H_CSR.getformat() == "csr"
    assert H_CSC.getformat() == "csc"

    # Check that the dense matrix has correct elements.
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

    # Test the error handling.
    with raises(Exception):
        H = system.matrix(format="blah")
    with raises(Exception):
        H = system.matrix(format=1)


def test_swave_hermitian():
    """Test that s-wave superconducting Hamiltonians are Hermitian.

    These Hamiltonians are constructed using the `swave()` function.
    """
    lattice = CubicLattice((10, 10, 1))
    system = Hamiltonian(lattice)

    σ_s = swave()
    with system as (H, Δ):
        for i in lattice.sites():
            H[i, i] = -0.1 * σ0
            Δ[i, i] = -0.1j * σ_s(i, i)  # On-site s-wave

        for i, j in lattice.bonds():
            H[i, j] = -1 * σ0
            Δ[i, j] = -0.3 * σ_s(i, j)  # Extended s-wave

    H = system._matrix.todense()
    assert np.allclose(H, H.T.conj())


def test_pwave_basic():
    """Brute-force test all p-wave superconducting order parameters.

    These have "d-vectors" of various kinds d(p) = e_i p_j, where it
    is known that if the d-vector points along e.g. p_x then the gap
    should only be non-zero for displacements along the x-axis.
    """
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
    """Check the p-wave superconductor property Δ(i, j) = -Δ(j, i)."""
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

        H = system._matrix.todense()
        assert np.allclose(H, H.T.conj())


def test_dwave_symmetries():
    """Brute-force test the behavior of the d_{x^2 - y^2} order parameter.

    This entails checking that the gap function has different signs
    along the x- and y-axes, and moreover that there is zero on-site
    or diagonal contributions to such a gap function.
    """
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

    H = system._matrix.todense()
    assert np.allclose(H, H.T.conj())


def test_ssd():
    """Test the mathematical properties of the "SSD" profile."""
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


def test_diagonalize():
    """Test that the matrix diagonalization works.

    Most importantly, we here validate that:
    - The defining equation H @ X_n = E_n @ X_n holds for each
      returned pair of eigenvalue E_n and eigenvector X_n;
    - That we only return positive eigenvalues ("Nambu doubling");
    - That the total number of eigenvalues is what we would expect.
    """
    # Instantiate a system with superconductivity and a magnetic barrier. Note
    # that magnetism breaks spin degeneracy, which is required for some tests.
    lattice = CubicLattice((10, 3, 2))
    system = Hamiltonian(lattice)

    with system as (H, Δ):
        for i in lattice.sites():
            H[i, i] = 4 * σ0
            if i[0] > 5:
                Δ[i, i] = 1 * jσ2
            elif i[0] > 3:
                H[i, i] = 6 * σ0 + 2 * σ3
        for i, j in lattice.bonds():
            H[i, j] = -1 * σ0

    # Calculate the eigenvalues the manual way.
    H = system.matrix(format="dense")
    E, X = eigh(H, subset_by_value=(0, np.inf))

    # If we ask for "raw" eigenvectors, this is what we want.
    E_raw, X_raw = system.diagonalize(format="raw")
    assert np.allclose(E, E_raw)
    assert np.allclose(X, X_raw)

    # Confirm that we got positive eigenvalues and that we have
    # interpreted the corresponding eigenvector matrix correctly.
    assert E.size == 2 * (10 * 3 * 2)
    for n, E_n in enumerate(E):
        assert E_n > 0
        assert np.allclose(H @ X[:, n], E_n * X[:, n])

    # Calculate the same eigenvalues via the package, and ensure
    # that the eigenvalues and eigenvectors are consistent.
    eigval, eigvec = system.diagonalize()
    assert np.allclose(eigval, E)
    for n, E_n in enumerate(E):
        for m in range(10 * 3 * 2):
            assert np.allclose(eigvec[n, m, 0], X[4 * m + 0, n])
            assert np.allclose(eigvec[n, m, 1], X[4 * m + 1, n])
            assert np.allclose(eigvec[n, m, 2], X[4 * m + 2, n])
            assert np.allclose(eigvec[n, m, 3], X[4 * m + 3, n])

    # Let's now test a different eigenvector format. This is the more intuitive
    # "wave function" format, which is supposed to to tell us the wave function
    # components at each lattice site corresponding to each particle species.
    # eig = system.diagonalize(format="wave")
    # for ε, (e_up, e_dn, h_up, h_dn) in eig.items():
    #     n = np.argwhere(eigval == ε)[0]
    #     assert np.allclose(ε, eigval[n])

    #     for coord in lattice.sites():
    #         i = lattice.index(coord)
    #         assert np.allclose(e_up[*coord], eigvec[n, i, 0])
    #         assert np.allclose(e_dn[*coord], eigvec[n, i, 1])
    #         assert np.allclose(h_up[*coord], eigvec[n, i, 2])
    #         assert np.allclose(h_dn[*coord], eigvec[n, i, 3])

    # Exception should be raised for "weird" eigenvector formats.
    with raises(Exception):
        system.diagonalize(format="foo")


def test_cuda():
    """Test that using CUDA support doesn't change results."""
    cupy = importorskip("cupy", reason="CuPy is not installed")

    # Instantiate a test system.
    lattice = CubicLattice((2, 3, 5))
    system = Hamiltonian(lattice)

    with system as (H, Δ):
        for i in lattice.sites():
            H[i, i] = 4 * σ0
            if i[0] == 0:
                Δ[i, i] = 1 * jσ2
            else:
                H[i, i] = 4 * σ0 + 1 * σ2
        for i, j in lattice.bonds():
            H[i, j] = -1 * σ0 + 2 * σ1

    # Solve with CPU and GPU.
    eigval_cpu, eigvec_cpu = system.diagonalize(format="raw", cuda=False)
    eigval_gpu, eigvec_gpu = system.diagonalize(format="raw", cuda=True)

    # Check that the same eigenvalues are found (but possibly in a different order).
    assert eigval_cpu.size == eigval_gpu.size
    assert np.allclose(np.sort(eigval_cpu), np.sort(eigval_gpu))

    # Check that the GPU eigenvectors satisfy their defining equation.
    H = system.matrix(format="dense")
    for n, E_n in enumerate(eigval_gpu):
        X_n = eigvec_gpu[..., n]
        assert np.allclose(H @ X_n, E_n * X_n)

    # Test that free energy vs. temperature is the same for GPU and CPU.
    for T in [0.001, 0.01, 0.1, 1.0]:
        F1 = system.free_energy(T, cuda=False)
        F2 = system.free_energy(T, cuda=True)
        assert np.allclose(F1, F2)


def test_free_energy():
    """Test that the free energy calculation works."""
    # Instantiate a simple S/N/F system.
    lattice = CubicLattice((10, 7, 3))
    system = Hamiltonian(lattice)

    with system as (H, Δ):
        for i in lattice.sites():
            if i[0] <= 3:
                H[i, i] = -0.5 * σ0
                Δ[i, i] = -1.0 * jσ2
            if i[0] >= 7:
                H[i, i] = +0.5 * σ0 + 1.5 * σ3

        for i, j in lattice.bonds():
            H[i, j] = -1 * σ0

    # Verify the expression for free energy.
    ε, χ = system.diagonalize()
    ε = np.hstack([-ε, +ε])
    for T in [0.01, 0.1, 1.0]:
        # Use the predefined function.
        F1 = system.free_energy(T)

        # Use standard expression.
        F2 = -(T / 2) * np.sum(np.log(1 + np.exp(-ε / T)))

        assert np.allclose(F1, F2)

    # Check the zero-temperature result.
    F1 = system.free_energy(0.0)
    F2 = (1 / 2) * np.sum(ε[ε < 0])
    assert np.allclose(F1, F2)

    # Check exception for negative energies.
    with raises(Exception):
        F = system.free_energy(-1.0)


def test_ldos():
    """Test that the local density of states (LDOS) works.

    Here, we primarily check that the LDOS has to be positive
    for every location and energy in every conceivable system.
    """
    # Instantiate a random metal.
    lattice = CubicLattice((5, 5, 2))
    system = Hamiltonian(lattice)

    with system as (H, Δ):
        for i in lattice.sites():
            H[i, i] = r() * σ0 + r() * σ1 + r() * σ2 + r() * σ3
            Δ[i, i] = (r() * σ1 + r() * σ2 + r() * σ3) @ jσ2

        for i, j in lattice.bonds():
            t = [r(), r(), r(), r()]
            H[i, j] = t[0] * σ0 + t[1] * σ1 + t[2] * σ2 + t[3] * σ3
            H[j, i] = t[0] * σ0 + t[1] * σ1 + t[2] * σ2 + t[3] * σ3
            Δ[i, j] = (r() * σ1 + r() * σ2 + r() * σ3) @ jσ2

        for i, j in lattice.edges():
            T = [r(), r(), r(), r()]
            H[i, j] = T[0] * σ0 + T[1] * σ1 + T[2] * σ2 + T[3] * σ3
            H[j, i] = T[0] * σ0 + T[1] * σ1 + T[2] * σ2 + T[3] * σ3
            Δ[i, j] = (r() * σ1 + r() * σ2 + r() * σ3) @ jσ2

    # Calculate the LDOS everywhere.
    sites = [(i, j, k) for i in range(5) for j in range(5) for k in range(2)]
    energies = [0.0, 0.01, 0.10, 0.50, 1.00, 2.00, 4.00]

    for site in sites:
        ρ = system.ldos(site, energies)
        assert all(ρ >= 0)
