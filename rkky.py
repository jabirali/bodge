#!/usr/bin/env python

from icecream import ic
from typer import run

from bodge import *


def main(
    sep: int,
    s1: str,
    s2: str,
    offset: int = 20,
    length: int = 64,
    width: int = 64,
    potential: float = -3.0,
    coupling: float = 3.0,
    supergap: float = 0.2,
    filename: str = "rkky.csv",
):
    """RKKY interaction between two impurities on a superconductor."""
    # Square lattice.
    lattice = CubicLattice((length, width, 1))
    ic(lattice.shape)

    # Impurity sites.
    x1 = offset
    y1 = width // 2
    z1 = 0

    x2 = x1 + sep
    y2 = y1
    z2 = z1

    if x2 <= x1 or x2 >= length - offset:
        raise RuntimeError("Offset requirements violated.")

    i1 = (x1, y1, z1)
    i2 = (x2, y2, z2)

    ic(i1)
    ic(i2)

    # Impurity spins.
    spins = {
        "x+": +σ1,
        "y+": +σ2,
        "z+": +σ3,
        "x-": -σ1,
        "y-": -σ2,
        "z-": -σ3,
    }

    S1 = spins[s1]
    S2 = spins[s2]

    ic(S1)
    ic(S2)

    # Construct the Hamiltonian.
    t = 1.0
    μ = potential
    J0 = coupling
    Δ0 = supergap
    Tc = Δ0 / 1.764

    system = Hamiltonian(lattice)
    with system as (H, Δ, _):
        for i in lattice.sites():
            Δ[i, i] = -Δ0 * jσ2
            if i == i1:
                H[i, i] = -μ * σ0 - (J0 / 2) * S1
            elif i == i2:
                H[i, i] = -μ * σ0 - (J0 / 2) * S2
            else:
                H[i, i] = -μ * σ0

        for i, j in lattice.bonds():
            H[i, j] = -t * σ0

    # Calculate the free energy.
    E = free_energy(system, 0.01 * Tc)

    # Save the results.
    with open(filename, "a+") as f:
        f.write(f"{s1}, {s2}, {sep}, {E}\n")


# Run main() when run as a script.
if __name__ == "__main__":
    run(main)
