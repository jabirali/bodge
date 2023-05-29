#!/usr/bin/env python

"""Calculation of critical temperature for s-wave superconductors with a magnetic defect."""

import numpy as np
from icecream import ic
from typer import run

from bodge import *

def main(delta: str):
    # Construct an appropriate lattice, including functions to determine
    # whether a particular region is superconducting or normal metallic.
    Lx, Ly, Lz = 64, 64, 2
    lattice = CubicLattice((Lx, Ly, Lz))

    def NM(i: Coord):
        x, y, z = i
        return z == 1

    def SC(i: Coord):
        x, y, z = i
        return z == 0 and x >= Lx // 4 and x < 3 * Lx // 4 and y >= Ly // 4 and y < 3 * Ly // 4

    def IN(i: Coord):
        return NM(i) or SC(i)

    # Load the interpolated profiles.
    with np.load(f"m_{delta}.npz") as f:
        mx, my, mz = f["mx"], f["my"], f["mz"]
        ic(mx, my, mz)
    
    # Define a function for magnetization at coordinates.
    def σ(i):
        x, y = i[:-1]
        return mx[x, y] * σ1 + my[x, y] * σ2 + mz[x, y] * σ3
    
    # Verify the profiles.
    import matplotlib.pyplot as plt
    import seaborn as sns

    xs, ys, mxs, mys, mzs = [], [], [], [], []
    for i in lattice.sites():
        xs.append(i[0])
        ys.append(i[1])
        mzs.append(np.real(np.trace(σ3 @ σ(i)) / 2))
        mys.append(np.real(np.trace(σ2 @ σ(i)) / 2))
        mxs.append(np.real(np.trace(σ1 @ σ(i)) / 2))
    
    sns.scatterplot(x=xs, y=ys, hue=mxs)
    plt.show()
    sns.scatterplot(x=xs, y=ys, hue=mys)
    plt.show()
    sns.scatterplot(x=xs, y=ys, hue=mzs)
    plt.show()

    # Perform Tc calculations.
    with open(f"scrit_m{delta}.dat", "w") as f:
        # Model parameters.
        t = 1.0
        μ = 0.5
        m = 2.0
        U = t

        # Calculate critical temperature.
        for τ in [0.03]:  # np.arange(0, 0.1, 0.01):
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
                for i, j in lattice.bonds(axis=2):
                    if IN(i) and IN(j):
                        H[i, j] = -τ * σ0

            Tc = critical_temperature(system, T_max=0.04)

            f.write(f"{sec}, {Tc}\n")
            f.flush()

if __name__ == "__main__":
    ic()
    run(main)
    ic()