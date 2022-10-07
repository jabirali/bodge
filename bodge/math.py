import warnings

import numpy as np

from .typing import *

# Fundamental constants.
π = np.pi

# Pauli matrices used to represent spin.
σ0 = np.array([[+1, 0], [0, +1]], dtype=np.complex128)
σ1 = np.array([[0, +1], [+1, 0]], dtype=np.complex128)
σ2 = np.array([[0, -1j], [+1j, 0]], dtype=np.complex128)
σ3 = np.array([[+1, 0], [0, -1]], dtype=np.complex128)

# Compact notation for imaginary versions.
jσ0 = 1j * σ0
jσ1 = 1j * σ1
jσ2 = 1j * σ2
jσ3 = 1j * σ3


def cheb_poly(X, I, N: int, R=None):
    """Chebyshev matrix polynomials T_n(X) for 0 ≤ n < N.

    The arguments X and I should be square matrices with the same dimensions,
    and these matrices can be either dense `np.array` or sparse `scipy.sparse`.

    Alternatively, you can divide the identity matrix I = [I_1, ..., I_K] into
    K columns and provide only one such block I_k as an argument to this function.
    This function will then calculate the corresponding blocks [T_n(X)]_k of the
    Chebyshev polynomials, which is useful for parallel construction of T_n(X).

    If the optional cutoff radius R is specified, the T_n(X) are projected
    onto the Local Krylov subspace spanned by {X^0, ..., X^R}, so that the
    T_n(X) for n > R are prevented from growing denser than T_R(X). This
    argument of course only makes sense when using sparse matrices.
    """

    # T_0(X) is simply equal to the provided identity matrix.
    T_0 = I
    yield T_0

    # T_1(X) reduces to X iff I is the full identity matrix.
    T_1 = X @ I
    yield T_1

    # T_n(X) is calculated via the Chebyshev recursion relation.
    for n in range(2, N):
        T_1, T_0 = 2 * (X @ T_1) - T_0, T_1

        # Local Krylov projection if a cutoff radius is specified.
        if R is not None:
            try:
                # Construct a Local Krylov subspace mask from the sparsity
                # structure of T_R(X), since this matrix contains all the
                # relevant contributions {X^0, ..., X^R} of the subspace.
                if n == R:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", np.ComplexWarning)
                        P = T_1.astype(dtype="int8", copy=False)
                        P.data[...] = 1

                # Project T_n(x) onto the Local Krylov subspace spanned by
                # elementwise multiplication by the mask constructed above.
                elif n > R:
                    T_1 = T_1.multiply(P)

            except AttributeError:
                raise ValueError("Cutoff radius is only supported for `scipy.sparse` matrices!")

        yield T_1


def cheb_coeff(F: Callable, N: int, odd=False, even=False):
    """Generate the Chebyshev coefficients for the given function.

    We define the coefficients f_n such that F(X) = ∑ f_n T_n(X) for any X,
    where the sum goes over 0 ≤ n < N and T_n(X) is found by `cheb_poly`.
    The optional arguments `odd` and `even` can be used to skip calculation
    of Chebyshev coefficients that are a priori known to be exactly zero.
    """
    # Calculate the ϕ_k such that x_k = cos(ϕ_k) are Chebyshev nodes.
    n = np.arange(N)
    ϕ = π * (n + 1 / 2) / N

    # Evaluate the provided function at the Chebyshev nodes x_k.
    f = np.array([F(x_k) for x_k in np.cos(ϕ)])

    # Perform the Chebyshev expansion.
    yield np.mean(f)
    for n in range(1, N):
        if odd and n % 2 == 0:
            yield 0
        elif even and n % 2 == 1:
            yield 0
        else:
            yield 2 * np.mean(f * np.cos(n * ϕ))


def cheb_kern(N: int):
    """Jackson kernel for preventing Gibbs oscillations in Chebyshev expansions.

    These factors g_n are used to calculate F(X) = ∑ f_n g_n T_n(X) for a
    finite number of terms 0 ≤ n < N. They provide a better approximation of
    F(X) than using an abrupt cutoff at n = N [equivalent to g_n = θ(N - n)].
    """
    Π = π / (N + 1)
    for n in range(N):
        yield (Π / π) * ((N - n + 1) * np.cos(Π * n) + np.sin(Π * n) / np.tan(Π))


def trace(X, N: int = 128):
    """Stochastic evaluation of the trace."""
    M = X.shape[-1]

    tr = 0
    for n in range(N):
        # Generate a Gaussian complex vector.
        v = np.random.randn(M, 2)
        v = (v[:, :1] + v[:, 1:] * 1j) / np.sqrt(2)

        # Stochastic evaluation of trace.
        tr += (v.T.conj() @ X @ v) / N

    return tr


def logdet(X, I, N: int = 128):
    """Stochastic Chebyshev evaluation of log det X.

    TODO: Replace I with a random vector, and integrate the stochastic trace.
    TODO: Create an exact version based on matrix diagonalization.
    """
    # Perform a Chebyshev expansion.
    fs = cheb_coeff(lambda x: np.log(1 - x), N)
    gs = cheb_kern(N)
    Ts = cheb_poly(I - X, I, N)

    # Calculate log det X.
    return sum(f * g * trace(T) for f, g, T in zip(fs, gs, Ts))
