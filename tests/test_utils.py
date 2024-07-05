from scipy.linalg import eigh

from bodge import *
from bodge.common import *


def test_diagonalize():
    # Instantiate a system with superconductivity and a barrier.
    lattice = CubicLattice((10, 10, 1))
    system = Hamiltonian(lattice)

    with system as (H, Δ):
        for i in lattice.sites():
            H[i, i] = 4 * σ0
            if i[0] > 5:
                Δ[i, i] = 1 * jσ2
            elif i[0] > 3:
                H[i, i] += 6 * σ0
        for i, j in lattice.bonds():
            H[i, j] = -1 * σ0

    # Calculate the eigenvalues the manual way.
    H = system(format="dense")
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
