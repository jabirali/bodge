#!/usr/bin/env python

from typing import Optional

import numpy as np
from typer import run

from bodge import *


def main(
    sep: int,
    s1: str,
    s2: str,
    gap_s: float,
    gap_p: float,
    dvector: str = "e_z * p_x",
    length: int = 51,
    width: int = 51,
    potential: float = -3.0,
    coupling: float = 3.0,
    winding: int = 0,
    filename: str = "rkky_sp.csv",
    cuda: bool = False,
):
    """RKKY interaction between two impurities on a superconductor."""

    # Square lattice.
    lattice = CubicLattice((length, width, 1))

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

    # Superconductivity.
    Δ_s = gap_s + 0.0j
    Δ_p = gap_p * 1.0j
    print(f":: {Δ_s}, {Δ_p}, {dvector}, {s1}, {s2}, {sep}\n")

    σ_s = jσ2
    σ_p = pwave(dvector)

    # Complex phase.
    def phase(i, j):
        x = (i[0] + j[0]) / 2
        L = lattice.shape[0]
        return np.exp(1j * 2 * np.pi * winding * x / L)

    for i in lattice.sites():
        if i[1] == 1:
            phi = phase(i, i)

    # Construct the Hamiltonian.
    t = 1.0
    μ = potential
    J0 = coupling

    system = Hamiltonian(lattice)
    with system as (H, Δ, _):
        for i in lattice.sites():
            Δ[i, i] = -Δ_s * σ_s * phase(i, i)
            if i == i1:
                H[i, i] = -μ * σ0 - (J0 / 2) * S1
            elif i == i2:
                H[i, i] = -μ * σ0 - (J0 / 2) * S2
            else:
                H[i, i] = -μ * σ0

        for i, j in lattice.bonds():
            H[i, j] = -t * σ0
            Δ[i, j] = -Δ_p * σ_p(i, j) * phase(i, j)

        for i, j in lattice.edges(axis=0):
            if j[0] == lattice.shape[0] - 1 and i[0] == 0:
                I = i
                J = (i[0] - 1, j[1], j[2])
            elif i[0] == lattice.shape[0] - 1 and j[0] == 0:
                J = j
                I = (j[0] - 1, i[1], i[2])
            else:
                raise RuntimeError("NOPE!")

            H[i, j] = -t * σ0
            Δ[i, j] = -Δ_p * σ_p(I, J) * phase(I, J)

    # Calculate the free energy.
    E = free_energy(system, 0.001 * t, cuda=cuda)

    # Save the results.
    with open(filename, "a+") as f:
        f.write(f'{Δ_s}, {Δ_p}, {winding}, {s1}, {s2}, {sep}, {E}, "periodic"\n')


if __name__ == "__main__":
    run(main)
