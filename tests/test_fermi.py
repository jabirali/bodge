from bodge import *
from bodge.common import *


def test_fermi_exact():
    """Test that the Fermi-Chebyshev expansion is analytically correct."""
    # Construct a Hamiltonian on a cubic lattice.
    lattice = CubicLattice((3, 5, 7))
    system = Hamiltonian(lattice)

    # For each lattice site, set the electron-electron part to (i/N) ∈ [0, +1].
    # Electron-hole symmetry produces corresponding diagonal terms in [-1, 0].
    with system as (H, Δ, _):
        for i in lattice.sites():
            H[i, i] = (lattice[i] / (lattice.size - 1)) * σ0

    # Extract the diagonal of the matrix above.
    d = system.matrix.diagonal()

    # Ensure that this diagonal spans the range [-1, +1].
    assert all(d >= -1)
    assert all(d <= +1)

    assert d.min() == -1
    assert d.max() == +1
    assert len(np.unique(d)) == 2 * lattice.size - 1

    # Perform the Fermi-Chebyshev expansion at a semi-random temperature.
    T = (1 + np.random.rand()) / 2
    fermi = FermiMatrix(system, 1024)

    F1 = fermi(T)
    f1 = F1.matrix.diagonal()
    print(f1)

    # Manually evaluate the Fermi function at the same temperature.
    f2 = 1 / (1 + np.exp(d / T))
    print(f2)

    # The two approaches to calculate the Fermi function are equivalent.
    assert np.allclose(f1, f2)
