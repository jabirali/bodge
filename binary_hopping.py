#!/usr/bin/env python3

"""Binary search in the inter-layer hopping parameter of an S/FM system."""

import logging as log
import sys

import numpy as np
import typer

from bodge import *


def main(tau: float, delta: str):
    # Construct an appropriate lattice, including functions to determine
    # whether a particular region is superconducting or normal metallic.
    Lx, Ly, Lz = 64, 64, 2
    lattice = CubicLattice((Lx, Ly, Lz))
    log.debug(f"Lattice: {lattice}")

    def NM(i: Coord):
        x, y, z = i
        return z == 1

    def SC(i: Coord):
        x, y, z = i
        return z == 0 and x >= Lx // 4 and x < 3 * Lx // 4 and y >= Ly // 4 and y < 3 * Ly // 4

    def IN(i: Coord):
        return NM(i) or SC(i)

    # Load the interpolated magnetization profiles.
    filename = f"m_{delta}.npz"
    log.debug(f"Filename: {filename}")
    with np.load(filename) as f:
        mx, my, mz = f["mx"], f["my"], f["mz"]

    # Generate spin matrices from the magnetization profiles.
    def σ(i):
        x, y = i[:-1]
        return mx[x, y] * σ1 + my[x, y] * σ2 + mz[x, y] * σ3

    # Model parameters.
    t = 1.0
    μ = 0.5
    m = 0.3
    U = t
    τ = tau
    log.info(f"Parameters: {t}, {τ}, {μ}, {m}, {U}")

    log.debug(f"Constructing the base Hamiltonian")
    system = Hamiltonian(lattice)
    with system as (H, Δ, V):
        for i in lattice.sites():
            # Chemical potential in non-empty space,
            # exchange field in non-superconductors.
            # Attractive Hubbard in superconductors.
            if NM(i):
                H[i, i] = -μ * σ0 - m * σ(i)
            if SC(i):
                H[i, i] = -μ * σ0
                V[i, i] = -U

        # Intra-plane hopping coefficient t.
        for i, j in lattice.bonds(axis=0):
            if IN(i) and IN(j):
                H[i, j] = -t * σ0
        for i, j in lattice.bonds(axis=1):
            if IN(i) and IN(j):
                H[i, j] = -t * σ0

    # Inter-plane hopping coefficient τ.
    log.debug(f"Updating inter-layer hopping term")
    with system as (H, Δ, V):
        for i, j in lattice.bonds(axis=2):
            if IN(i) and IN(j):
                H[i, j] = -τ * σ0


if __name__ == "__main__":
    log.basicConfig(
        filename=__file__.replace(".py", ".log"),
        filemode="w",
        format="\x1b[32m%(asctime)s \x1b[33m%(levelname)s \x1b[0m%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=log.DEBUG,
    )
    log.info(f"Executing: {sys.argv}")
    typer.run(main)
