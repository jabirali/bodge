import numpy as np
import pytest

from bodge.fermi import chebyshev


def test_chebyshev():
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

        # Verify that the iterator terminates correctly.
        with pytest.raises(StopIteration) as e:
            next(chebs)
