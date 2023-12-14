#!/usr/bin/env python

"""Binary search in the inter-layer hopping parameter of an S/FM system."""

import logging as log
import sys

import numpy as np
import typer
from tqdm import trange

from bodge import *


def main(delta: str):
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
    μ = 0.5 * t
    m = 0.3 * t
    U = 1.0 * t

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

    log.debug(f"Constructing the Fermi matrix")
    fermi = FermiMatrix(system, order=1200)

    # Binary search procedure.
    T_c = 0.01 * t
    τ_min = 0.0 * t
    τ_max = 1.0 * t
    Δ_init = 1e-4

    for n in trange(20, unit="τ", smoothing=0):
        # Update current hopping parameter.
        τ_now = (τ_min + τ_max) / 2
        log.info(f"Inter-layer hopping: {τ_now}")
        with system as (H, Δ, V):
            for i, j in lattice.bonds(axis=2):
                if IN(i) and IN(j):
                    H[i, j] = -τ_now * σ0

        # Self-consistency iterations.
        Δ_now = {i: Δ_init for i in lattice.sites()}
        for m in trange(8, unit="Δ", smoothing=0):
            with system as (H, Δ, V):
                for i in lattice.sites():
                    if (i, i) in V:
                        Δ[i, i] = Δ_now[i] * jσ2
            Δ_now = fermi(T_c).order_swave()
            log.debug(f"Mean gap: {np.mean(np.abs(Δ_now))}")

        # Update hopping interval based on gap change.
        Δ_fin = np.median(np.abs(Δ_now[np.nonzero(Δ_now)]))
        if Δ_fin > Δ_init:
            τ_min = τ_now
        else:
            τ_max = τ_now

    τ_now = (τ_min + τ_max) / 2
    log.info(f"Obtained the final value: τ = {τ_now} t.")


if __name__ == "__main__":
    log.basicConfig(
        filename=__file__.replace(".py", ".log"),
        format="\x1b[32m%(asctime)s \x1b[33m%(levelname)s \x1b[0m%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=log.DEBUG,
    )

    args = " ".join(sys.argv)
    log.info(f"\x1b[31;1m{args}\x1b[0m")
    typer.run(main)
