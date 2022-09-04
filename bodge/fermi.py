import numpy as np

from .consts import *
from .hamiltonian import Hamiltonian
from .typing import *


def chebyshev(X, I, N):
    """Calculate the Chebyshev matrix polynomials T_n(M)."""

    T_0 = I
    T_n = X

    for n in range(1, N):
        T_n, T_0 = 2 * X @ T_n - T_0, T_n

    return T_n
