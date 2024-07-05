"""Collection of numerical algorithms that can be used on `Hamiltonian` instances."""

import scipy.linalg as la
from tqdm import tqdm

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


@typecheck
def ldos(
    system: Hamiltonian, site: Coord, energies: Matrix, resolution=None
) -> tuple[Matrix, Matrix]:
    """Calculate the local density of states via a resolvent operator approach.

    We define the resolvent operator as [(ε+iη)I - H] R(ε) = I, which can be
    divided into vectors R(ε) = [r_1 ... r_N] and I = [e_1 ... e_N]. Diagonal
    elements of the resolvent matrix correspond to the density of states. By
    calculating one vector at a time via a sparse linear solver `spsolve`, the
    local density of states can be efficiently calculated at specific points.

    Note that you need only provide a list of positive energies at which to
    calculate the density of states, since electron-hole symmetry is used to
    get the negative-energy solutions in an efficient manner.

    The algorithm implemented here is explained in detail in Appendix A of:

    Ouassou et al. PRB 109, 174506 (2024).
    DOI: 10.1103/PhysRevB.109.174506
    """
    import scipy.sparse.linalg as sla

    # Prepare input and output variables.
    H, M, I = system(format="csc")
    results = []

    # Adaptive energy resolution.
    εs = np.sort(energies)

    if resolution is not None:
        ηs = resolution * np.repeat(1, len(εs))
    elif len(εs) > 1:
        ηs = np.gradient(εs)
    else:
        raise ValueError("Resolution required for single-energy calculations.")

    ωs = εs + 1j * ηs

    # Construct a reduced identity matrix with only these indices.
    N = H.shape[1]
    M = 4

    i = system.lattice[site]
    rows = np.array([4 * i + α for α in range(4)])
    cols = np.arange(M)
    data = np.repeat(1, M)

    B = CscMatrix((data, (rows, cols)), shape=(N, M))

    # Calculate the density of states.
    results = {}
    for ω in tqdm(ωs, unit="ε", desc="LDOS"):
        # Solve the linear equations for the resolvent.
        A = ω * I - H
        X = sla.spsolve(A, B)

        # Extract the few elements of interest.
        x = X.multiply(B).sum(axis=0)

        # Calculate and store the density of states.
        e_up = x[0, 0]
        e_dn = x[0, 1]
        h_up = x[0, 2]
        h_dn = x[0, 3]

        results[+np.real(ω)] = -np.imag(e_up + e_dn) / π
        results[-np.real(ω)] = -np.imag(h_up + h_dn) / π

    ωs = np.array(results.keys())
    dos = np.array(results.values())

    return ωs, dos


def free_energy(system: Hamiltonian, temperature: float = 0.0, cuda=False) -> float:
    """Calculate the Landau free energy for a given Hamiltonian.

    This is done by computing all the positive eigenvalues ε_n of the matrix,
    and subsequently evaluating the entropy contributions to the free energy.
    The resulting free energy is then formulated as F = U - TS, where U is
    the internal energy (calculated from ɛ_n), T is the provided temperature,
    and S is the system's entropy (calculated from ε_n and T).

    The argument `cuda` can be used to turn on GPU calculations, which
    can be several orders of magnitude faster if you have an NVIDIA GPU.

    Note that in general, U should also have a constant contribution, which
    corresponds to the non-matrix parts to the Hamiltonian operator. These
    implicitly depend on e.g. all mean fields in the system. We can neglect
    these contributions when doing non-selfconsistent calculations. But if you
    want to calculate e.g. the critical temperature, then it is important that
    you calculate this constant and add it to the return value of this function.

    The algorithm implemented here is explained in Appendix C of:

    Ouassou et al. PRB 109, 174506 (2024).
    DOI: 10.1103/PhysRevB.109.174506

    """
    T = temperature
    H = system(format="dense")

    # Calculate the eigenvalues via a dense parallel algorithm. If the
    # argument `cuda` is set, use a GPU instead of a CPU. Note that we
    # only calculate the eigenvalues here, which is in general much
    # faster than calculating the eigenvectors (as in `diagonalize`).
    if cuda:
        import cupy as cp
        import cupy.linalg as cla

        ε = cp.asnumpy(cla.eigvalsh(cp.asarray(H)))
    else:
        ε = la.eigvalsh(H)

    # Extract the positive eigenvalues.
    ε = ε[ε > 0]

    # Internal energy.
    U = -(1 / 2) * np.sum(ε)

    # Entropy contribution.
    if T == 0:
        S = 0
    elif T > 0:
        S = np.sum(np.log(1 + np.exp(-ε / T)))
    else:
        raise ValueError("Expected non-negative temperature!")

    # Free energy
    F = U - T * S

    return F
