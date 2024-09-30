import numpy as np

from .common import *


@typecheck
def singlet(
    eigs: dict[float, tuple[Matrix, Matrix, Matrix, Matrix]],
    potential: float,
    temperature: float = 1e-3,
):
    """Calculate the singlet order parameter Δ.

    The first input argument is the eigenvalues and eigenvectors of
    the system in the "wave" format, containing electron wave functions.

    NOTE: Under development and may contain bugs...

    """
    U = potential
    T = temperature

    # Fermi-Dirac distribution.
    def f(E):
        return (1 - np.tanh(E / (2 * T))) / 2

    # Calculation
    gap = 0
    for E, (e_up, e_dn, h_up, h_dn) in eigs.items():
        gap += U * e_dn * h_up.conj() * f(+E)
        gap += U * e_up * h_dn.conj() * f(-E)

    return gap
