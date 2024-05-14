#!/usr/bin/env python

"""Calculate the local density of states around a magnetic impurity.

This is useful to e.g. determine the YSR bound states that might exist
in such materials, which is likely related to RKKY oscillations.
"""

# %% Imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bodge import *
from bodge.utils import ldos, pwave
from scipy.signal import find_peaks

# %% Parameters used in paper.
t = 1.0
μ = -3.0 * t
Δ0 = 0.1 * t
J0 = 3.0 * t

# %% Construct a 2D lattice.
# Lx = 200
# Ly = 200
Lx = 800
Ly = 800
Lz = 1
print(Lx, Ly, Lz)

lattice = CubicLattice((Lx, Ly, 1))


# %% Function to perform calculations.
def main(ds, δs):
    dfs = []
    for d in ds:
        # Convert string to d-vector.
        d_ = pwave(d)

        # Construct the base Hamiltonian.
        system = Hamiltonian(lattice)
        with system as (H, Δ, _):
            for i in lattice.sites():
                H[i, i] = -μ * σ0
            for i, j in lattice.bonds():
                H[i, j] = -t * σ0
                Δ[i, j] = -Δ0 * d_(i, j)

        # Prepare calculation points.
        energies = np.linspace(0.00, 2 * Δ0, 51)
        i = (Lx // 2, Ly // 2, 0)
        sites = [(i[0] + δ, i[1], i[2]) for δ in δs]

        # Calculate the density of states without impurity.
        df0 = ldos(system, sites, energies)
        df0["d"] = d
        df0["imp"] = False

        print(df0)
        dfs.append(df0)

        # Add the impurities to the Hamiltonian.
        with system as (H, Δ, V):
            H[i, i] = -μ * σ0 - (J0 / 2) * σ3

        # Calculate the density of states with impurity.
        df1 = ldos(system, sites, energies)
        df1["d"] = d
        df1["imp"] = True

        print(df1)
        dfs.append(df1)

        df = pd.concat(dfs)
        save(df)

    return df


def save(df, name="ysr3.csv"):
    df["δ"] = df["x"] - Lx // 2
    df.to_csv(name, columns=["d", "δ", "imp", "ε", "dos"], index=False)


# %% Test the function above.
# ds = ["e_x * p_x"]
# δs = [1, 2]
# df = main(ds, δs)
# # display(df)
# save(df, 'test.csv')

# %% Perform actual calculations.
ds = ["e_z * (p_x + jp_y)"]
δs = [1, 2, 3, 4, 5]
df = main(ds, δs)
# save(df)
