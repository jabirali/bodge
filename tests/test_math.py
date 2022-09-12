import numpy as np
import pytest
from scipy.sparse import bsr_matrix, csr_matrix

from bodge.math import *


def test_pauli():
    # Test that the quaternion identities hold.
    assert np.allclose(σ1 @ σ1, σ0)
    assert np.allclose(σ2 @ σ2, σ0)
    assert np.allclose(σ3 @ σ3, σ0)

    assert np.allclose(σ1 @ σ2, jσ3)
    assert np.allclose(σ2 @ σ3, jσ1)
    assert np.allclose(σ3 @ σ1, jσ2)

    assert np.allclose(σ1 @ σ2 @ σ3, jσ0)


def test_chebyshev_diag():
    """Test that Chebyshev polynomials satisfy T_n(x) = cos[n acos x]."""
    # Construct diagonal matrices with elements in [-1, +1].
    I = np.identity(11)
    x = np.linspace(+1, -1, 11)
    X = np.diag(x)

    # Perform analytical and numerical Chebyshev expansions.
    N = 13
    with np.printoptions(precision=2, suppress=True):
        # Construct the numerical Chebyshev iterator.
        chebs = chebyshev(X, I, N)

        # Verify that each Chebyshev polynomial is correct.
        for n in range(0, N):
            Tn_bench = np.diag(next(chebs))
            Tn_exact = np.cos(n * np.arccos(x))

            print(f"Chebyshev polynomial T_{n}(x):")
            print(Tn_bench)
            print(Tn_exact)

            assert np.allclose(Tn_bench, Tn_exact)

        # Verify that the iterator terminates after N terms.
        with pytest.raises(StopIteration) as e:
            next(chebs)


def test_chebyshev_blocks():
    """Test that Chebyshev blocks satisfy [T_n1(X) ... T_nK(X)] = T_n(X).

    Here, T_nk(X) are blocks of the Chebyshev polynomials calculated
    from identity matrix blocks I_k, while T_n(X) are the Chebyshev
    polynomials calculated using the complete identity matrix I.
    This ensures that we can do parallel blockwise computations.
    """
    # Construct an identity matrix and a random matrix.
    I = np.identity(17)
    X = np.random.randn(17, 17)

    # Divide the identity matrix into blocks.
    I_1 = I[:, :4]
    I_2 = I[:, 4:10]
    I_3 = I[:, 10:]

    assert np.allclose(np.hstack([I_1, I_2, I_3]), I)

    # Construct Chebyshev iterators.
    chebs_0 = chebyshev(X, I, 13)

    chebs_1 = chebyshev(X, I_1, 13)
    chebs_2 = chebyshev(X, I_2, 13)
    chebs_3 = chebyshev(X, I_3, 13)

    # Verify the reconstruction of T_n(X) from blocks.
    for n, (Tn_0, Tn_1, Tn_2, Tn_3) in enumerate(zip(chebs_0, chebs_1, chebs_2, chebs_3)):
        Tn = np.hstack([Tn_1, Tn_2, Tn_3])

        print(f"Difference for T_{n}(X):")
        print(Tn - Tn_0)

        assert np.allclose(Tn, Tn_0)


def test_chebyshev_sparse():
    """Test that dense and sparse computations are equivalent."""
    I1 = np.identity(16)
    X1 = np.random.randn(16, 16)

    I2 = bsr_matrix(I1, blocksize=(4, 4))
    X2 = bsr_matrix(X1, blocksize=(4, 4))

    chebs_1 = chebyshev(X1, I1, 10)
    chebs_2 = chebyshev(X2, I2, 10)

    for n, (T_n1, T_n2) in enumerate(zip(chebs_1, chebs_2)):
        assert np.allclose(T_n1, T_n2.todense())


def test_chebyshev_radius():
    """Test the Local Krylov expansion feature."""
    # Construct a realistic tridiagonal matrix X.
    I = np.identity(64)
    X = I.copy()
    for n in range(64 - 2):
        X[n, n + 2] = -1 / 2
        X[n + 2, n] = -1 / 2

    # Convert the above into sparse matrices.
    X = csr_matrix(X)
    I = csr_matrix(I)

    # Construct the relevant Chebyshev generators.
    R = 5
    cheb_1 = chebyshev(X, I, 2 * R + 1)
    cheb_2 = chebyshev(X, I, 2 * R + 1, R)

    # Generate the T_n(X) with and without cutoff.
    for n, (Tn_1, Tn_2) in enumerate(zip(cheb_1, cheb_2)):
        # The first R matrices should be exactly the same.
        if n <= R:
            assert Tn_1.nnz == Tn_2.nnz
            assert Tn_1.nnz == (X**n).nnz
            assert np.allclose(Tn_1.todense(), Tn_2.todense())

        # The next matrix is the first with a cutoff. The *stored* elements in
        # the two matrices should be exactly the same at this point.
        elif n == R + 1:
            assert Tn_1.nnz > Tn_2.nnz
            assert (Tn_1 - Tn_2).nnz == (X ** (R + 1)).nnz - (X**R).nnz

        # The last matrices should scale like X^n or X^R, respectively.
        else:
            assert Tn_1.nnz > Tn_2.nnz
            assert Tn_1.nnz == (X**n).nnz
            assert Tn_2.nnz == (X**R).nnz


def test_fermi_expansion():
    """Test that the Chebyshev expansion of the Fermi function is analytically correct."""
    # Diagonal matrices with elements in [-1, +1].
    M = 71
    I = np.identity(M)
    x = np.linspace(+1, -1, M)
    X = np.diag(x)

    # Chebyshev expand the Fermi function f(ε).
    N = 200
    fs = fermi(0.05, N)
    Ts = chebyshev(X, I, N)

    f1 = np.zeros(M)
    for f, T in zip(fs, Ts):
        f1 += np.diag(f * T)

    # Calculate the Fermi function manually.
    f2 = 1 / (1 + np.exp(x / 0.05))

    # The two approaches should be identical.
    assert np.allclose(f1, f2)
