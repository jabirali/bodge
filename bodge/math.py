import warnings
from math import ceil

import multiprocess as mp
import numpy as np
import scipy.sparse as sps
from tqdm import trange

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


def cheb(F, X, S, N, filter: Optional[Callable] = None, preserve=True, pbar=True) -> bsr_matrix:
    """Parallelized Chebyshev expansion using Kernel Polynomial Method (KPM)."""
    # Coefficients for the kernel polynomial method.
    f = cheb_coeff(F, N)
    g = cheb_kern(N)
    c = [f_n * g_n for f_n, g_n in zip(f, g)]

    # Prepare the index filter function.
    if filter is None:
        filter = lambda _: True

    # Blockwise calculation of the Chebyshev expansion.
    W = 32
    K = ceil(X.shape[1] / W)

    def kernel(k):
        # Identity block.
        I_k = idblk(block=k, blocksize=W, dim=X.shape[0])
        if I_k is None:
            return None

        # Structure block.
        # TODO: Rename to mask.
        if preserve:
            S_k = S @ I_k
            # S_k = X @ I_k
            # with warnings.catch_warnings():
            #     warnings.simplefilter("ignore", np.ComplexWarning)
            #     mask = np.sum(np.abs(S_k.data), axis=(1, 2)) != 0
            #     S_k.data[mask, ...] = 1
            #     S_k = bsr_matrix(S_k, blocksize=(4, 4), dtype=np.int8)
        else:
            S_k = 1

        # Chebyshev expansion.
        T_k = cheb_poly(X, I_k, N)
        F_k = 0j
        for n, (c_n, T_kn) in enumerate(zip(c, T_k)):
            if filter(n):
                F_k += c_n * T_kn.multiply(S_k)

        return F_k

    # Parallel execution of the blockwise calculation.
    with mp.Pool() as pool:
        if pbar:
            blocks = trange(K, unit="blk", desc="kpm", smoothing=0, leave=False)
        else:
            blocks = range(K)

        F = pool.map(kernel, blocks, chunksize=1)

    # TODO: Add COO matrices instead of stacking BSR matrices?
    return sps.hstack([F_k for F_k in F if F_k is not None]).tobsr((4, 4))


def cheb_poly(X, I, N: int):
    """Chebyshev matrix polynomials T_n(X) for 0 ≤ n < N.

    The arguments X and I should be square matrices with the same dimensions,
    and these matrices can be either dense `np.array` or sparse `scipy.sparse`.

    Alternatively, you can divide the identity matrix I = [I_1, ..., I_K] into
    K columns and provide only one such block I_k as an argument to this function.
    This function will then calculate the corresponding blocks [T_n(X)]_k of the
    Chebyshev polynomials, which is useful for parallel construction of T_n(X).

    TODO: Update doscstring below.

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

        yield T_1


def cheb_coeff(F: Callable, N: int):
    """Generate the Chebyshev coefficients for the given function.

    We define the coefficients f_n such that F(X) = ∑ f_n T_n(X) for any X,
    where the sum goes over 0 ≤ n < N and T_n(X) is found by `cheb_poly`.

    The optional argument `filter` can be set to a function that
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


def idblk(block, blocksize, dim):
    """Partition the identity matrix into column blocks."""
    # TODO: Generate from a given i-list.
    # Determine blocksize and offset for this block.
    offset = block * blocksize

    # Blocksize correction for the last blocks in the batch.
    if offset >= dim:
        return None
    elif offset + blocksize >= dim:
        blocksize = dim - offset

    # Create the corresponding block of the identity matrix.
    shape = (dim, blocksize)
    diag = np.repeat(np.int8(1), blocksize)
    matrix = sps.dia_matrix((diag, [-offset]), shape, dtype=np.int8)

    # Return each identity block.
    return matrix.tobsr((4, 4))
