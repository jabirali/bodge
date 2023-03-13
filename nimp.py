#!/usr/bin/env python

from typing import Optional

from icecream import ic
from typer import run

from bodge import *


def main(
    s: str,
    x: int,
    y: int,
    length: int = 64,
    width: int = 64,
    potential: float = -3.0,
    coupling: float = 3.0,
    filename: str = "impurity.csv",
):
    """Free energy for a single impurity."""

    # Square lattice.
    lattice = CubicLattice((length, width, 1))
    ic(lattice.shape)

    # Impurity site.
    z = 0
    i0 = (x, y, z)

    if x < 0 or x >= length or y < 0 or y >= width:
        raise RuntimeError("Offset requirements violated.")

    ic(i0)

    # Impurity spins.
    spins = {
        "0": 0.0 * σ0,
        "x+": +σ1,
        "y+": +σ2,
        "z+": +σ3,
        "x-": -σ1,
        "y-": -σ2,
        "z-": -σ3,
    }

    S0 = spins[s]

    ic(S0)

    # Construct the Hamiltonian.
    t = 1.0
    μ = potential
    J0 = coupling

    system = Hamiltonian(lattice)

    with system as (H, Δ, _):
        for i in lattice.sites():
            if i == i0:
                H[i, i] = -μ * σ0 - (J0 / 2) * S0
            else:
                H[i, i] = -μ * σ0

        for i, j in lattice.bonds():
            H[i, j] = -t * σ0

    # Calculate the free energy.
    E = free_energy(system, 0.001)

    # Save the results.
    with open(filename, "a+") as f:
        f.write(f"{width}, {length}, {x}, {y}, {s}, {E}\n")


if __name__ == "__main__":
    ic()
    run(main)
    ic()
