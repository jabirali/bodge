from pytest import CallInfo

from .common import *
from .hamiltonian import Hamiltonian
from .math import *


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

    TODO:
    - Factor out the Green function calculation as separate function that
      accepts Hamiltonian indices as arguments. Can be used for e.g. currents.
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
        X = sa.spsolve(A, B)

        # Extract the few elements of interest.
        x = X.multiply(B).sum(axis=0)

        # Calculate and store the density of states.
        # TODO: Reconstruct and trace 2x2 matrices.
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
    H = system(format="dense")
    eigval, eigvec = la.eigh(H, subset_by_value=(0.0, np.inf), overwrite_a=True, driver="evr")

    # Restructure the eigenvectors to have the format eigvec[n, i, α],
    # where n corresponds to eigenvalue E[n], i is a position index, and
    # α represents the combined particle and spin index {e↑, e↓, h↑, h↓}.
    eigvec = eigvec.T.reshape((eigval.size, -1, 4))

    return eigval, eigvec


def spectral(system: Hamiltonian, energies, resolution: float = 1e-3) -> list[Matrix]:
    """Calculate the exact spectral function of the system via direct inversion.

    Note that this method is quite inefficient since it uses dense matrices;
    it is meant as a benchmark, not for actual large-scale calculations.
    """
    # Restore the Hamiltonian scale and switch to dense matrices.
    H = system(format="dense")

    # The resolution is controlled by the imaginary energy.
    η = resolution * 1j

    # Calculate the spectral function via direct inversion.
    spectral = []
    for ω in energies:
        Gᴿ = la.inv((ω + η) * I - H)
        Gᴬ = la.inv((ω - η) * I - H)
        A = (Gᴿ - Gᴬ) / (-2j * π)

        spectral.append(A)

    return spectral


def free_energy(system: Hamiltonian, temperature: float = 0.01):
    """Calculate the Landau free energy from a given Hamiltonian matrix.

    This is done by computing all the positive eigenvalues ε_n of the matrix,
    and subsequently evaluating the entropy contributions to the free energy.
    """
    T = temperature
    H = system(format="dense")

    # Calculate the eigenvalues via a dense parallel algorithm.
    ε = la.eigh(H, overwrite_a=True, eigvals_only=True, driver="evr")

    # Extract the positive eigenvalues.
    ε = ε[ε > 0]

    # TODO: Mean field contributions.
    E0 = 0.0

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


def pwave(desc: str):
    """Convert a d-vector expression into a p-wave gap function."""
    # Basis vectors for spin axes.
    e_x = np.array([[1], [0], [0]])
    e_y = np.array([[0], [1], [0]])
    e_z = np.array([[0], [0], [1]])

    je_x = 1j * e_x
    je_y = 1j * e_y
    je_z = 1j * e_z

    # Basis vectors for momentum.
    p_x = e_x.T
    p_y = e_y.T
    p_z = e_z.T

    jp_x = 1j * p_x
    jp_y = 1j * p_y
    jp_z = 1j * p_z

    # Convert the d-vector expression to a 3x3 numerical matrix.
    D = eval(desc)

    # Construct gap matrix Δ(p) = [d(p)⋅σ] jσ2 = [(D'p) ⋅ σ] jσ2.
    # In practice, we do this by calculating Δ = D'σ jσ2, such
    # that we simply end up with the gap matrix Δ(p) = Δ ⋅ p.
    Δ = np.einsum("kp,kab,bc -> pac", D, σ, jσ2) / 2

    # Function for evaluating Δ(p) on the lattice.
    def Δ_p(i: Coord, j: Coord) -> Matrix:
        δ = np.subtract(j, i)
        return np.einsum("iab,i -> ab", Δ, δ)

    return Δ_p
