import scipy.linalg as sla
from scipy.sparse.linalg import spsolve

from .hamiltonian import *
from .math import *
from .typing import *


def ldos(system, sites, energies, resolution):
    """Calculate the local density of states via a resolvent operator approach.

    We define the resolvent operator as [(ε+iη)I - H] R(ε) = I, which can be
    divided into vectors R(ε) = [r_1 ... r_N] and I = [e_1 ... e_N]. Diagonal
    elements of the resolvent matrix correspond to the density of states. By
    calculating one vector at a time via a sparse linear solver `spsolve`, the
    local density of states can be efficiently calculated at specific points.
    """
    dos = {}

    H = system.compile()
    I = system.identity.tocsr()
    ε = [energy / system.scale for energy in energies]
    η = resolution / system.scale

    for n, ε_n in enumerate(ε):
        A = (ε_n + η * 1j) * I - H

        # M = A + 0.1j * I
        # ilu = spilu(M)
        # Mx = lambda x: ilu.solve(x)
        # M = LinearOperator(A.shape, Mx)

        for i in sites:
            n = 4 * system.lattice[i]
            e_n = csr_matrix(([1], ([n], [0])), shape=(H.shape[1], 1))

            # x_n = e_n.copy()
            # e_n = np.zeros((H.shape[1], 1))
            # e_n[n] = 1

            # print(A.shape)
            # print(e_n.shape)

            # x_n, err = cg(A, e_n.todense(), x_n, tol=1e-3)
            # x_n, err = cg(A, e_n.todense(), tol=1e-6, M=M)
            # x_n, err = gmres(A, e_n.todense(), tol=1e-2)
            # x_n, err = qmr(A, e_n.todense(), tol=1e-2)
            # x_n, err = gcrotmk(A, e_n.todense(), tol=1e-2)
            # print(err)

            x_n = spsolve(A, e_n)

            dos[i, ε_n] = -np.imag(x_n[n]) / (π * system.scale)
            print(dos[i, ε_n])

    return dos


def diagonalize(system: Hamiltonian) -> tuple[Array, Array]:
    """Calculate the exact eigenstates of the system via direct diagonalization.

    This calculates the eigenvalues and eigenvectors of the system. Due to
    the particle-hole symmetry, only positive eigenvalues are calculated.
    Note that this method is inefficient since it uses dense matrices; it
    is meant as a benchmark, not for actual large-scale calculations.
    """
    # Calculate the relevant eigenvalues and eigenvectors.
    H = system.scale * system.matrix.todense()
    eigval, eigvec = eigh(H, subset_by_value=(0.0, np.inf), overwrite_a=True, driver="evr")

    # Restructure the eigenvectors to have the format eigvec[n, i, α],
    # where n corresponds to eigenvalue E[n], i is a position index, and
    # α represents the combined particle and spin index {e↑, e↓, h↑, h↓}.
    eigvec = eigvec.T.reshape((eigval.size, -1, 4))

    return eigval, eigvec


def spectral(system: Hamiltonian, energies: ArrayLike, resolution: float = 1e-3) -> list[Array]:
    """Calculate the exact spectral function of the system via direct inversion.

    Note that this method is quite inefficient since it uses dense matrices;
    it is meant as a benchmark, not for actual large-scale calculations.
    """
    # Restore the Hamiltonian scale and switch to dense matrices.
    H = system.scale * system.matrix.todense()
    I = system.identity.todense()

    # The resolution is controlled by the imaginary energy.
    η = resolution * 1j

    # Calculate the spectral function via direct inversion.
    spectral = []
    for ω in energies:
        Gᴿ = inv((ω + η) * I - H)
        Gᴬ = inv((ω - η) * I - H)
        A = (Gᴿ - Gᴬ) / (-2j * π)

        spectral.append(A)

    return spectral


def energy(system: Hamiltonian):
    """Calculate the free energy for a given physical system.

    This is done by computing all the positive eigenvalues ε_n of the matrix,
    and evaluating the entropy contributions ∑ log[f(ε_n)] to the free energy.

    TODO: Incorporate superconducting contributions of the type |Δ|^2/V.
    """

    H = system.scale * system.matrix.todense()
    ε = sla.eigh(H, overwrite_a=True, eigvals_only=True, driver="evr")

    # TODO: Calculate the actual free energy from this.
    return np.array(sorted(ε_n for ε_n in ε if ε_n > 0))


def dvector(desc: str):
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
    Δ = np.einsum("kp,kab,bc -> pac", D, σ, jσ2)

    # Function for evaluating Δ(p) on the lattice.
    def Δ_p(i: Coord, j: Coord):
        δ = np.subtract(j, i)
        return np.einsum("iab,i -> ab", Δ, δ)

    return Δ_p
