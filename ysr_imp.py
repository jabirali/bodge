#!/usr/bin/env python

"""Calculate the local density of states around a magnetic impurity.

This is useful to e.g. determine the YSR bound states that might exist
in such materials, which is likely related to RKKY oscillations.
"""

import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

from bodge import *
from bodge.utils import ldos

# Construct a 2D lattice.
Lx = 80
Ly = 80
Lz = 1

lattice = CubicLattice((Lx, Ly, 1))

# Loop over tight-binding params.
t = 1
for Δ0 in [0.2]:
    for J0 in [0.5]:
        for μ in [-0.9]:
            for δ in [0, 1, 2, 3, 4, 5]:
                # Construct the base Hamiltonian.
                system = Hamiltonian(lattice)
                with system as (H, Δ, V):
                    for i in lattice.sites():
                        H[i, i] = -μ * σ0
                        Δ[i, i] = -Δ0 * jσ2
                    for i, j in lattice.bonds():
                        H[i, j] = -t * σ0

                # Prepare calculation points.
                ii = (Lx//2, Ly//2, 0)
                sites = [(ii[0] + δ, ii[1], ii[2])]
                energies = np.linspace(0.00, 2*Δ0, 101)

                # Add the single impurity.
                with system as (H, Δ, V):
                    H[ii, ii] = -μ * σ0 - J0 * σ3

                # Calculate the density of states.
                df = ldos(system, sites, energies)

                plt.figure()
                plt.plot(df.ε / Δ0, df.dos)
                plt.xlabel(r"Energy $\epsilon/\Delta$")
                plt.ylabel(r"LDOS $N(\epsilon, x)$")
                plt.title(rf"Distance $\delta = {δ}$ along $x$-axis from impurity")
                plt.show()

                # Look for potential YSR DOS peaks.
                df = df[df.ε >= 0]
                ε = np.array(df.ε)
                dos = np.array(df.dos)
                n_ysr, _ = find_peaks(dos)

                # print(dos)
                # print(Δ0, J0, μ, δ, ε[n_ysr], dos[n_ysr])