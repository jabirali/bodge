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


def chebyshev(X, I, N):
    """Iterator that calculates the Chebyshev matrix polynomials T_n(X).

    If the matrix X has dimensions M×M and I is a corresponding identity
    matrix, we generate Chebyshev polynomials T_n(X) of order n ∈ [0, N].

    Alternatively, you can divide the identity matrix into M×(M/K) blocks
    I = [I_1 ... I_K], in which case `chebyshev(X, I_k, N)` yields the
    corresponding blocks [T_n(X)]_k of the Chebyshev polynomials T_n(X).
    This is useful for algorithms computing matrix blocks in parallel.
    """
    # T_0(X) is simply equal to the provided identity matrix.
    T_0 = I
    yield T_0

    # T_1(X) reduces to X if I is the full identity matrix.
    T_1 = X @ I
    yield T_1

    # T_n(X) is calculated via a recursion relation.
    for n in range(2, N):
        T_1, T_0 = 2 * X @ T_1 - T_0, T_1
        yield T_1
