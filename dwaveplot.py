#!/usr/bin/env python

"""Calculate the RKKY interaction for s- and d-wave superconductors."""

from typing import Optional

import numpy as np
import pandas as pd
from icecream import ic
from typer import run

from bodge import *


def main(
    length: int = 128,
    width: int = 128,
    potential: float = -3.0,
    coupling: float = 3.0,
    supergap: float = 0.10,
    filename: str = "ldos_sd.csv",
):
    """RKKY interaction between two impurities on a superconductor."""

    # Square lattice
    lattice = CubicLattice((length, width, 1))
    ic(lattice.shape)

    # Superconductivity.
    σ_s = swave()
    σ_d = dwave()

    ic(σ_s)
    ic(σ_d((2, 2, 0), (3, 2, 0)))
    ic(σ_d((2, 2, 0), (1, 2, 0)))
    ic(σ_d((2, 2, 0), (2, 3, 0)))
    ic(σ_d((2, 2, 0), (2, 1, 0)))

    # Construct the Hamiltonian.
    t = 1.0
    μ = potential
    J0 = coupling
    Δd = supergap * 1.0
    Δs = supergap * 1.0j

    E = np.linspace(1e-16, 0.3 * t, int(t / 0.02))

    # D+iS wave
    system = Hamiltonian(lattice)
    with system as (H, Δ, _):
        for i in lattice.sites():
            Δ[i, i] = -Δs * σ_s
            H[i, i] = -μ * σ0

        for i, j in lattice.bonds():
            H[i, j] = -t * σ0
            Δ[i, j] = -Δd * σ_d(i, j)

    ds = ldos(system=system, sites=[(length // 2, width // 2, 0)], energies=E)

    # D wave
    system = Hamiltonian(lattice)
    with system as (H, Δ, _):
        for i in lattice.sites():
            H[i, i] = -μ * σ0

        for i, j in lattice.bonds():
            H[i, j] = -t * σ0
            Δ[i, j] = -Δd * σ_d(i, j)

    d = ldos(system=system, sites=[(length // 2, width // 2, 0)], energies=E, resolution=Δd / 30)

    # S wave
    system = Hamiltonian(lattice)
    with system as (H, Δ, _):
        for i in lattice.sites():
            Δ[i, i] = -Δs * σ_s
            H[i, i] = -μ * σ0

        for i, j in lattice.bonds():
            H[i, j] = -t * σ0

    s = ldos(system=system, sites=[(length // 2, width // 2, 0)], energies=E, resolution=Δd / 30)

    # Merge and save
    df = ds.copy()
    df["d+is"] = ds["dos"]
    df["d"] = d["dos"]
    df["s"] = s["dos"]

    df.to_csv(filename)


if __name__ == "__main__":
    ic()
    run(main)
    ic()
