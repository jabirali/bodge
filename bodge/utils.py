"""Collection of numerical algorithms that can be used on `Hamiltonian` instances."""

import scipy.linalg as la

from .common import *
from .hamiltonian import Hamiltonian


def diagonalize(system: Hamiltonian) -> tuple[Matrix, Matrix]:
    """Calculate the exact eigenstates of the system via direct diagonalization.

    This calculates and returns the eigenvalues and eigenvectors of the system.
    If you run it as `E, v = diagonalize(system)`, then the eigenvalue `E[n]`
    corresponds to the eigenvector `v[n, :, :]`, where the remaining indices
    of the vector correspond to a position index and a Nambu⊗Spin index.
    """
    # Calculate the relevant eigenvalues and eigenvectors.
    # TODO: Add CUDA support here via `cupy.linalg.eigh`.
    H = system(format="dense")
    eigval, eigvec = la.eigh(H, subset_by_value=(0.0, np.inf), overwrite_a=True, driver="evr")

    # Restructure the eigenvectors to have the format eigvec[n, i, α],
    # where n corresponds to eigenvalue E[n], i is a position index, and
    # α represents the combined particle and spin index {e↑, e↓, h↑, h↓}.
    eigvec = eigvec.T.reshape((eigval.size, -1, 4))

    return eigval, eigvec
