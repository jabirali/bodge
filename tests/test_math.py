import pytest
from scipy.stats import unitary_group

from bodge.common import *
from bodge.chebyshev import *


def test_pauli():
    # Test that the quaternion identities hold.
    assert np.allclose(σ1 @ σ1, σ0)
    assert np.allclose(σ2 @ σ2, σ0)
    assert np.allclose(σ3 @ σ3, σ0)

    assert np.allclose(σ1 @ σ2, jσ3)
    assert np.allclose(σ2 @ σ3, jσ1)
    assert np.allclose(σ3 @ σ1, jσ2)

    assert np.allclose(σ1 @ σ2 @ σ3, jσ0)


def test_chebyshev_exact():
    """Test that Chebyshev polynomials satisfy T_n(x) = cos[n acos x]."""
    # Construct diagonal matrices with elements in [-1, +1].
    I = np.identity(11)
    x = np.linspace(+1, -1, 11)
    X = np.diag(x)

    # Perform analytical and numerical Chebyshev expansions.
    N = 13
    with np.printoptions(precision=2, suppress=True):
        # Construct the numerical Chebyshev iterator.
        chebs = cheb_poly(X, I, N)

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
    chebs_0 = cheb_poly(X, I, 13)

    chebs_1 = cheb_poly(X, I_1, 13)
    chebs_2 = cheb_poly(X, I_2, 13)
    chebs_3 = cheb_poly(X, I_3, 13)

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

    I2 = BsrMatrix(I1, blocksize=(4, 4))
    X2 = BsrMatrix(X1, blocksize=(4, 4))

    chebs_1 = cheb_poly(X1, I1, 10)
    chebs_2 = cheb_poly(X2, I2, 10)

    for n, (T_n1, T_n2) in enumerate(zip(chebs_1, chebs_2)):
        assert np.allclose(T_n1, T_n2.todense())


def test_chebyshev_unitary():
    """Test that Chebyshev polynomials satisfy U† T_n(D) U = T_n(X)."""
    # Diagonal matrices with elements in [-1, +1].
    M = 13
    I = np.identity(M)
    D = np.diag(np.linspace(+1, -1, M))

    # Unitary transformation with a matrix U.
    U = unitary_group.rvs(M)
    Ut = U.T.conj()

    # Chebyshev expansion using non-diagonal vs. diagonal matrices.
    Tx = cheb_poly(U @ D @ Ut, I, 10)
    Td = cheb_poly(D, I, 10)

    # Check the bespoke mathematical properties.
    for Tx_n, Td_n in zip(Tx, Td):
        assert np.allclose(Tx_n, U @ Td_n @ Ut)


def test_jackson_kernel():
    """Test that the Jackson kernel behaves as a reasonable regularization."""
    # Prepare one generator with N = 100 and one with N = 1,000,000.
    jackson_small = cheb_kern(int(1e2))
    jackson_large = cheb_kern(int(1e6))

    # Check that the generator with N = 100 tapers off more quickly than the
    # one with N = 1,000,000, while decreasing monotonically from one to zero.
    g_s0 = 1
    for g_s1, g_l1 in zip(jackson_small, jackson_large):
        assert g_s1 <= g_s0
        assert g_s1 <= g_l1
        assert g_s1 >= 0

        g_s0 = g_s1

    # Check that g_n ≈ 0 for n = N but that g_n ≈ 1 for n ≪ N.
    assert np.allclose(g_s1, 0, atol=1e-4)
    assert np.allclose(g_l1, 1, atol=1e-4)


def test_idblk():
    """Test that identity blocks stack to an identity matrix."""
    N = 4 * 13
    M = 4 * 3
    K = 6

    Is = [idblk(k, M, N) for k in range(K)]
    I1 = sp.hstack([I_k for I_k in Is if I_k is not None])
    I2 = sp.identity(N)
    assert (I1.todense() == I2.todense()).all()
