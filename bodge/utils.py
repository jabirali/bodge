"""Collection of numerical algorithms that can be used on `Hamiltonian` instances."""

import pandas as pd
import scipy.linalg as la
import scipy.sparse.linalg as sla
from tqdm import tqdm

from .common import *
from .hamiltonian import Hamiltonian


def ldos(system, sites, energies, resolution=None) -> pd.DataFrame:
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

    TODO:
    - Factor out the Green function calculation as separate function that
      accepts Hamiltonian indices as arguments. Can be used for e.g. currents.
    - Generalize the LDOS calculation to trace over 2x2 matrices. This would
      allow us to easily extract the spin-resolved LDOS in the future.
    """
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
    M = 4 * len(sites)

    rows = np.array([4 * system.lattice[i] + α for i in sites for α in range(4)])
    cols = np.arange(M)
    data = np.repeat(1, M)

    B = CscMatrix((data, (rows, cols)), shape=(N, M))

    # Calculate the density of states.
    for ω in tqdm(ωs, unit="ε", desc="LDOS"):
        # Solve the linear equations for the resolvent.
        A = ω * I - H
        X = spla.spsolve(A, B)

        # Extract the few elements of interest.
        x = X.multiply(B).sum(axis=0)

        # Calculate and store the density of states.
        for n, i in enumerate(sites):
            e_up = x[0, 4 * n + 0]
            e_dn = x[0, 4 * n + 1]
            h_up = x[0, 4 * n + 2]
            h_dn = x[0, 4 * n + 3]

            e_dos = -np.imag(e_up + e_dn) / π
            h_dos = -np.imag(h_up + h_dn) / π

            ε = np.real(ω)
            results.append(
                pd.DataFrame.from_dict({"x": i[0], "y": i[1], "z": i[2], "ε": +ε, "dos": [e_dos]})
            )
            results.append(
                pd.DataFrame.from_dict({"x": i[0], "y": i[1], "z": i[2], "ε": -ε, "dos": [h_dos]})
            )

    # Merge the dataframes and return.
    return pd.concat(results).sort_values(by=["x", "y", "z", "ε"])


def diagonalize(system: Hamiltonian) -> tuple[Matrix, Matrix]:
    """Calculate the exact eigenstates of the system via direct diagonalization.

    This calculates the eigenvalues and eigenvectors of the system. Due to
    the particle-hole symmetry, only positive eigenvalues are calculated.
    Note that this method is inefficient since it uses dense matrices; it
    is meant as a benchmark, not for actual large-scale calculations.
    """
    # Calculate the relevant eigenvalues and eigenvectors.
    # TODO: Integrate CUDA implementation here as well.
    H = system(format="dense")
    eigval, eigvec = la.eigh(H, subset_by_value=(0.0, np.inf), overwrite_a=True, driver="evr")

    # Restructure the eigenvectors to have the format eigvec[n, i, α],
    # where n corresponds to eigenvalue E[n], i is a position index, and
    # α represents the combined particle and spin index {e↑, e↓, h↑, h↓}.
    eigvec = eigvec.T.reshape((eigval.size, -1, 4))

    return eigval, eigvec


def free_energy(system: Hamiltonian, temperature: float = 0.01, constant: float = 0.0) -> float:
    """Calculate the Landau free energy for a given Hamiltonian.

    This is done by computing all the positive eigenvalues ε_n of the matrix,
    and subsequently evaluating the entropy contributions to the free energy.
    The resulting free energy is then formulated as F = U - TS, where U is
    the internal energy (calculated from ɛ_n), T is the provided temperature,
    and S is the system's entropy (calculated from ε_n and T).

    Note that in general, U should also have a constant contribution, which
    corresponds to the non-matrix parts to the Hamiltonian operator. These
    implicitly depend on e.g. all mean fields in the system. We can neglect
    these contributions when doing non-selfconsistent calculations. But if you
    want to calculate e.g. the critical temperature, then it is important that
    you calculate this `constant` yourself and provide it as an argument.

    The algorithm implemented here is explained in Appendix C of:

    Ouassou et al. PRB 109, 174506 (2024).
    DOI: 10.1103/PhysRevB.109.174506
    """
    T = temperature
    E0 = constant
    H = system(format="dense")

    # Calculate the eigenvalues via a dense parallel algorithm. My benchmarks
    # have shown that this is usually faster than using the sparse solver.
    # TODO: Re-integrate the CUDA implementation from master branch.
    ε = la.eigh(H, overwrite_a=True, eigvals_only=True, driver="evr")

    # Extract the positive eigenvalues.
    ε = ε[ε > 0]

    # Internal energy.
    U = E0 - (1 / 2) * np.sum(ε)

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
