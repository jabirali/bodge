#!/usr/bin/env python

"""RKKY interactions between two magnetic impurities on superconductor."""
from click import argument, command, option
from icecream import ic

from bodge import *


@command()
@option("--lattice-length", "Lx", default=81, type=int)
@option("--lattice-width", "Ly", default=81, type=int)
@option("--chemical-potential", "μ0", default=-3.0, type=float)
@option("--superconding-gap", "Δ0", default=0.2, type=float)
@option("--impurity-coupling", "J0", default=3.0, type=float)
@option("--spin-left", "s1", default="+z", type=str)
@option("--spin-right", "s2", default="+z", type=str)
@option("--external-distance", "d1", default=20, type=int)
@option("--internal-distance", "d2", default=20, type=int)
def rkky(Lx, Ly, μ0, Δ0, J0, s1, s2, d1, d2):
    # Square lattice.
    lattice = CubicLattice((Lx, Ly, 1))
    ic(lattice.shape)

    # Impurity sites.
    x1 = d1
    y1 = Ly // 2
    z1 = 0

    x2 = d1 + d2 + 1
    y2 = y1
    z2 = z1

    if x2 > Lx - d1:
        raise RuntimeError("External distance requirements violated.")

    i1 = (x1, y1, z1)
    i2 = (x2, y2, z2)

    ic(i1)
    ic(i2)

    # Impurity spins.
    spins = {
        "+x": +σ1,
        "+y": +σ2,
        "+z": +σ3,
        "-x": -σ1,
        "-y": -σ2,
        "-z": -σ3,
    }

    S1 = spins[s1]
    S2 = spins[s2]

    ic(S1)
    ic(S2)

    # Construct the Hamiltonian.
    system = Hamiltonian(lattice)
    with system as (H, Δ, _):
        for i in lattice.sites():
            Δ[i, i] = -Δ0 * jσ2
            if i == i1:
                H[i, i] = -μ0 * σ0 - J0 * S1
            elif i == i2:
                H[i, i] = -μ0 * σ0 - J0 * S2
            else:
                H[i, i] = -μ0 * σ0

        for i, j in lattice.bonds():
            H[i, j] = -σ0

    # Critical temperature.
    Tc = Δ0 / 1.764

    # Calculate the free energy.
    F = free_energy(system, 0.01 * Tc)

    # Return the results.
    return (s1, s2, d2, F)


if __name__ == "__main__":
    results = rkky()
    with open("rkky.csv", "a+") as f:
        f.write(",".join(results))
