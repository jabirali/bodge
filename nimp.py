#!/usr/bin/env python

from typing import Optional

from bodge import *
from icecream import ic
from typer import run


def main(
    s: str,
    x: int,
    y: int,
    dvector: Optional[str] = None,
    length: int = 80,
    width: int = 80,
    potential: float = -3.0,
    coupling: float = 3.0,
    supergap: float = 0.00,
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

    # Superconductivity.
    ic(dvector)
    if dvector is None:
        # s-wave only.
        σ_s = jσ2
        σ_p = None

        ic(σ_s)
        ic(σ_p)
    else:
        # p-wave only.
        σ_s = None
        σ_p = pwave(dvector)

        ic(σ_s)
        ic(σ_p((2, 2, 0), (3, 2, 0)))
        ic(σ_p((2, 2, 0), (1, 2, 0)))
        ic(σ_p((2, 2, 0), (2, 3, 0)))
        ic(σ_p((2, 2, 0), (2, 1, 0)))

    # Construct the Hamiltonian.
    t = 1.0
    μ = potential
    J0 = coupling
    Δ0 = supergap
    Tc = Δ0 / 1.764

    system = Hamiltonian(lattice)

    with system as (H, Δ, _):
        for i in lattice.sites():
            if σ_s is not None:
                Δ[i, i] = -Δ0 * σ_s
            if i == i0:
                H[i, i] = -μ * σ0 - (J0 / 2) * S0
            else:
                H[i, i] = -μ * σ0

        for i, j in lattice.bonds():
            H[i, j] = -t * σ0
            if σ_p is not None:
                Δ[i, j] = -Δ0 * σ_p(i, j)

    # Calculate the free energy.
    E = free_energy(system, 0.01 * Tc)

    # Save the results.
    with open(filename, "a+") as f:
        f.write(f"{dvector}, {x}, {y}, {s}, {E}\n")


if __name__ == "__main__":
    ic()
    run(main)
    ic()
