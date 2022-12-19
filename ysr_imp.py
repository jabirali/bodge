#!/usr/bin/env python

"""Calculate the local density of states around a magnetic impurity.

This is useful to e.g. determine the YSR bound states that might exist
in such materials, which is likely related to RKKY oscillations.
"""

#%% Imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from bodge import *
from bodge.utils import ldos, pwave

# Construct a 2D lattice.
Lx = 64
Ly = 64
Lz = 1

lattice = CubicLattice((Lx, Ly, 1))

# d-vector for p-wave superconductivity.
# for s in ["e_z * p_x"]:
for s in ["e_z * p_x", "e_z * p_y", "e_z * (p_x + jp_y)", "(e_x + je_y) * (p_x + jp_y) / 2"]:
    print(s)
    d = pwave(s)
    # d = pwave("e_z * (p_x + jp_y)")
    # d = pwave("e_z * p_y")
    # d = pwave("(e_x + je_y) * (p_x + jp_y) / 2")

    # Loop over tight-binding params.
    t = 1
    for Δ0 in [0.1]:
        for J0 in [1.5]:
            for μ in [-2.5]:
                for δ in [1]:  # [0, 1, 2, 3, 4, 5]:
                    # NOTE: Not optimal, use many sites instead.
                    # Construct the base Hamiltonian.
                    system = Hamiltonian(lattice)
                    with system as (H, Δ, V):
                        for i in lattice.sites():
                            H[i, i] = -μ * σ0
                            # Δ[i, i] = -Δ0 * jσ2
                        for i, j in lattice.bonds():
                            H[i, j] = -t * σ0
                            Δ[i, j] = -Δ0 * d(i, j)

                    # Prepare calculation points.
                    ii = (Lx // 2, Ly // 2, 0)
                    sites = [(ii[0] + δ, ii[1], ii[2])]
                    energies = np.linspace(0.00, 2 * Δ0, 51)

                    # Calculate the density of states without impurity.
                    df0 = ldos(system, sites, energies)

                    # Add the single impurity.
                    with system as (H, Δ, V):
                        H[ii, ii] = -μ * σ0 - J0 * σ3

                    # Calculate the density of states with impurity.
                    df1 = ldos(system, sites, energies)

                    plt.figure()
                    # plt.plot(df0.ε / Δ0, df0.dos, 'k')
                    plt.plot(df0.ε / Δ0, df0.dos, df1.ε / Δ0, df1.dos)
                    plt.xlabel(r"Energy $\epsilon/\Delta$")
                    plt.ylabel(r"LDOS $N(\epsilon, x)$")
                    plt.title(rf"Distance $\delta = {δ}$ along $x$-axis from impurity")
                    plt.show()

                    # Look for potential YSR DOS peaks.
                    # df1 = df1[df1.ε >= 0]
                    # ε = np.array(df1.ε)
                    # dos = np.array(df1.dos)
                    # n_ysr, _ = find_peaks(dos)

                    # print(dos)
                    # print(Δ0, J0, μ, δ, ε[n_ysr], dos[n_ysr])

# %%
