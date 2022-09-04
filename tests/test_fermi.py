import numpy as np

from bodge.fermi import chebyshev


def test_chebyshev():
    """Test that Chebyshev polynomials satisfy T_n(x) = cos[n acos x]."""
    # Construct diagonal matrices with elements in [-1, +1].
    I = np.identity(11)
    x = np.linspace(+1, -1, 11)
    X = np.diag(x)

    # Compare the numerical and exact Chebyshev expansions.
    N = 13
    with np.printoptions(precision=2, suppress=True):
        for n in range(1, 13):
            Tn_exact = np.cos(n * np.arccos(x))
            Tn_bench = np.diag(chebyshev(X, I, n))

            print(f"Chebyshev polynomial T_{n}(x):")
            print(Tn_bench)
            print(Tn_exact)

            assert np.allclose(Tn_bench, Tn_exact)
