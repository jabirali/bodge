from .chebyshev import *
from .common import *
from .fermi import FermiMatrix
from .hamiltonian import Hamiltonian
from .lattice import Lattice


def deform(system: Hamiltonian, method="ssd"):
    """Interface to Hamiltonian deformation techniques."""
    if method == "ssd":
        return lambda i, j: ssd(system.lattice, i, j)
    else:
        raise RuntimeError("Method not implemented")


def ssd(lattice: Lattice, i: Coord, j: Coord):
    """Profile used for a sine-squared deformation on a cubic lattice."""
    # Determine the origin and maximum radius for the system. Since we use
    # corner-centered coordinates, these two values are actually the same.
    # Note however the offset to get a maximum coordinate from a shape.
    R = np.array([(N - 1) for N in lattice.shape]) / 2

    # Determine the distance vectors of the provided coordinates
    # i = (i_x, i_y, i_z), j = (j_x, j_y, j_z) from the origin.
    r_i = np.array(i) - R
    r_j = np.array(j) - R
    r = (r_i + r_j) / 2

    # We now consider only the magnitude of the distances.
    R = la.norm(R)
    r = la.norm(r)

    # Calculate the sine-squared deformation.
    return (1 / 2) * (1 + np.cos(π * r / (R + 1 / 2)))


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


def critical_temperature(
    system: Hamiltonian,
    order: int = 1200,
    bisects: int = 12,
    iters: int = 8,
    T_min: float = 0.0,
    T_max: float = 1.0,
) -> float:
    """Calculate the critical temperature using a bisection method."""
    # Prepare the Fermi matrix expansion.
    lattice = system.lattice
    fermi = FermiMatrix(system, order)

    # Determine critical temperature via binary search.
    Δ_init = 1e-5
    T_now = (T_min + T_max) / 2

    print("Determining critical temperature:")
    for n in trange(bisects, unit="temp", smoothing=0):
        # Gap initialization.
        Δ_now = {i: Δ_init for i in lattice.sites()}

        # Self-consistency iterations.
        for m in trange(iters, unit="gap", smoothing=0):
            with system as (H, Δ, V):
                for i in lattice.sites():
                    if (i, i) in V:
                        Δ[i, i] = Δ_now[i] * jσ2
            Δ_now = fermi(T_now).order_swave()

        # Update critical temperature estimate based on
        # whether the gap at this temperature increased.
        Δ_fin = np.median(np.abs(Δ_now[np.nonzero(Δ_now)]))
        if Δ_fin > Δ_init:
            T_min = T_now
        else:
            T_max = T_now
        T_now = (T_min + T_max) / 2

        # print(f"Tc({n}):\t{T_now}")

    return T_now


def zero_gap(
    system: Hamiltonian,
    order: int = 1200,
    bisects: int = 12,
    iters: int = 8,
    Δ_min: float = 0.0,
    Δ_max: float = 1.0,
) -> float:
    """Calculate the zero-temperature gap using a bisection method."""
    # Prepare the Fermi matrix expansion.
    lattice = system.lattice
    fermi = FermiMatrix(system, order)

    # Determine critical temperature via binary search.
    Δ_now = (Δ_min + Δ_max) / 2
    T_now = 1e-3

    print("Determining critical temperature:")
    for n in trange(bisects, unit="gap", smoothing=0):
        # Gap initialization.
        Δ_map = {i: Δ_now for i in lattice.sites()}

        # Self-consistency iterations.
        for m in trange(iters, unit="gap", smoothing=0):
            with system as (H, Δ, V):
                for i in lattice.sites():
                    if (i, i) in V:
                        Δ[i, i] = Δ_map[i] * jσ2
            Δ_map = fermi(T_now).order_swave()

        # Update gap estimate based on whether the change.
        Δ_fin = np.median(np.abs(Δ_map[np.nonzero(Δ_map)]))
        if Δ_fin > Δ_now:
            Δ_min = Δ_now
        else:
            Δ_max = Δ_now
        Δ_now = (Δ_min + Δ_max) / 2

        print(f"Δ({n}):\t{Δ_now}")

    return Δ_now


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
    I = np.identity(H.shape[0])

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


def swave() -> Matrix:
    """Generate the s-wave superconducting order parameter.

    This is only implemented for consistency with `pwave` and `dwave`.
    Moreover, since the structure is the same for on-site and extended
    s-wave orders, it is not a function of coordinates but just a matrix.
    """

    return jσ2


def pwave(desc: str):
    """Generate the p-wave superconducting order parameter.

    You should provide a d-vector [e.g. "(p_x + jp_y) * (e_x + je_y)"] in order
    to construct a p-wave triplet order parameter. The function then returns a
    new function Δ_p(i, j) which lets you evaluate the superconducting order
    parameter for two lattice sites with coordinates i and j. This is useful
    when constructing the Hamiltonian of the system numerically.
    """
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
        # Nearest-neighbor vector.
        δ = np.subtract(j, i)

        # The above vector is normalized such that |δ| = 1 on a square lattice.
        # But this is not the case when we turn on periodic boundary conditions,
        # where i = (200, 0, 0) and j = (0, 0, 0) might have answer |j - i| = 1
        # when periodic boundary conditions are considered. We handle this here.
        if np.sum(np.abs(δ)) > 1:
            # Wrap around the x axis.
            if δ[0] > +1 and δ[1] == 0 and δ[2] == 0:
                δ[0] = -1
            elif δ[0] < -1 and δ[1] == 0 and δ[2] == 0:
                δ[0] = +1
            # Wrap around the y axis.
            elif δ[1] > +1 and δ[2] == 0 and δ[0] == 0:
                δ[1] = -1
            elif δ[1] < -1 and δ[2] == 0 and δ[0] == 0:
                δ[1] = +1
            # Wrap around the z axis.
            elif δ[2] > +1 and δ[0] == 0 and δ[1] == 0:
                δ[2] = -1
            elif δ[2] < -1 and δ[0] == 0 and δ[1] == 0:
                δ[2] = +1
            # Verify that the situation is resolved.
            else:
                raise RuntimeError(f"Invalid δ = {δ} encountered!")

        return np.einsum("iab,i -> ab", Δ, δ)

    return Δ_p


def dwave():
    """Generate the d-wave superconducting order parameter.

    This function returns a function Δ(i, j) that takes two lattice sites i, j,
    and returns the superconducting order parameter Δ between those two sites.

    We specifically consider the d_{x^2 - y^2} order parameter on a presumably
    square lattice. This means that the order parameter should have a momentum
    structure ~ (p_x^2 - p_y^2)/p_F^2 and spin structure ~ jσ_2 (spin-singlet).
    It might work on non-square lattices as well, but this has to be verified.
    """

    def Δ_d(i: Coord, j: Coord) -> Matrix:
        δ = np.subtract(j, i)
        Δ_ij = (δ[0] ** 2 - δ[1] ** 2) / (np.sum(δ**2) + 1e-16)

        return Δ_ij * jσ2

    return Δ_d
