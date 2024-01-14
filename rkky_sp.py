#!/usr/bin/env python


from typing import Optional

from icecream import ic
from typer import run

from bodge import *


def main(
    sep: int,
    s1: str,
    s2: str,
    mix: float, # Δ_s vs. Δ_p
    phase: bool,  # s+p vs. s+ip
    dvector: Optional[str] = None,
    length: int = 100,
    width: int = 100,
    potential: float = -3.0,
    coupling: float = 3.0,
    supergap: float = 0.10,
    filename: str = "rkky_sp.csv",
):
    """RKKY interaction between two impurities on a superconductor."""

    # Square lattice.
    lattice = CubicLattice((length, width, 1))
    ic(lattice.shape)

    # Impurity sites.
    x1 = length // 2
    y1 = width // 2
    z1 = 0

    x2 = x1 + sep
    y2 = y1
    z2 = z1

    if x2 <= x1 or x2 >= length:
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

    # Superconductivity.
    Δ_s = supergap
    Δ_p = supergap * mix
    if phase:
        Δ_p *= 1.0j
    print(f":: {Δ_s}, {Δ_p}, {dvector}, {s1}, {s2}, {sep}\n")

    σ_s = jσ2
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

    system = Hamiltonian(lattice)
    with system as (H, Δ, _):
        for i in lattice.sites():
            Δ[i, i] = -Δ_s * σ_s
            if i == i1:
                H[i, i] = -μ * σ0 - (J0 / 2) * S1
            elif i == i2:
                H[i, i] = -μ * σ0 - (J0 / 2) * S2
            else:
                H[i, i] = -μ * σ0

        for i, j in lattice.bonds():
            H[i, j] = -t * σ0
            Δ[i, j] = -Δ_p * σ_p(i, j)

    # Calculate the free energy.
    E = free_energy(system, 0.001 * t)

    # Save the results.
    with open(filename, "a+") as f:
        f.write(f"{Δ_s}, {Δ_p}, {dvector}, {s1}, {s2}, {sep}, {E}\n")


if __name__ == "__main__":
    ic()
    run(main)
    ic()
