import numpy as np

from .consts import *
from .hamiltonian import Hamiltonian
from .typing import *


def chebyshev(X, I, N):
    """Calculate Chebyshev matrix polynomials T_n(X) of order n ∈ [0, N]."""
    T_0 = I
    yield T_0

    T_1 = X
    yield T_1

    for n in range(2, N):
        T_1, T_0 = 2 * X @ T_1 - T_0, T_1
        yield T_1
