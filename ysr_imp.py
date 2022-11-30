#!/usr/bin/env python

"""Calculate the local density of states around a magnetic impurity.

This is useful to e.g. determine the YSR bound states that might exist
in such materials, which is likely related to RKKY oscillations.
"""

import pandas as pd
import numpy as np

from bodge import *
from bodge.utils import ldos

# Tight-binding parameters in eV taken from PRB 104 094527 (2021).
# t = 0.2
# μ = -0.6
# J0 = 1.0e-3
# Δ0 = 1.5e-3
t = 1
Δ0 = 0.1

# Construct a 2D lattice.
Lx = 80
Ly = 80
Lz = 1

for J0 in reversed([*map(lambda x: x*Δ0, [2.0, 1.5, 1.1, 1.0, 0.9, 0.5, 0.2, 0.01])]):
    lattice = CubicLattice((Lx, Ly, 1))

    # Construct the base Hamiltonian.
    system = Hamiltonian(lattice)
    with system as (H, Δ, V):
        # Superconductivity.
        for i in lattice.sites():
            Δ[i, i] = -Δ0 * jσ2

        # Hopping terms
        for i, j in lattice.bonds():
            H[i, j] = -t * σ0

    # Prepare calculation points.
    ii = (Lx//2, Ly//2, 0)
    sites = [(ii[0] + 1, ii[1], ii[2])]
    energies = np.linspace(0.00, 2*Δ0, 101)

    # Calculate the YSR LDOS.
    dfs = []
    for μ in [0.0]: #np.linspace(-4, 4, 3):
        # Update the chemical potential.
        with system as (H, Δ, V):
            for i in lattice.sites():
                H[i, i] = -μ * σ0

        df0 = ldos(system, sites, energies)
        df0 = df0[(df0.ε >= 0) & (df0.ε < Δ0)]

        # Add the impurity contribution.
        with system as (H, Δ, V):
            H[ii, ii] = -μ * σ0 -J0 * σ3

        df1 = ldos(system, sites, energies)
        df1 = df1[(df1.ε >= 0) & (df1.ε < Δ0)]

        # df['μ'] = μ

        # Determine 
        # df = df[df.ε > 0]
        ε = np.array(df0.ε)
        dos = np.array(df1.dos - df0.dos)
        ddos = np.gradient(np.gradient(dos))
        print(dos)

        # NOTE: This algo is wrong; we want derivative to change sign.
        n_ysr = np.argmin(ddos)
        print(ddos)
        if n_ysr > 0 and n_ysr < len(ddos) - 1:
            ε_ysr = ε[n_ysr]
        else:
            ε_ysr = np.nan

        print(J0, ε_ysr, dos[n_ysr], ddos[n_ysr])

        # Save and print results.

    #     print(np.diff(np.array(df[df.ε>=0].dos)))
    #     dfs.append(df)

    # # Merge and save dataframesk.
    # df = pd.concat(dfs, ignore_index=True)
    # df.to_csv('ysr_swave.csv')
