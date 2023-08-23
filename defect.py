#!/usr/bin/env python

"""Critical temperature for s-wave superconductors with a magnetic defect.

The scenarios considered here correspond to a superconductor in contact
with a magnetic insulator, such that the electron flow inside the magnet
can be neglected. This way, we can approximate it as an effective exchange
field inside the superconductor (with a corresponding magnetic texture).
"""

import numpy as np
from icecream import ic
from typer import run

from bodge import *

def main(delta: str, mag: float):
    ic(delta)
    ic(mag)

    # Construct a square lattice.
    Lx, Ly, Lz = 32, 32, 1
    lattice = CubicLattice((Lx, Ly, Lz))

    # Load the interpolated profiles.
    δ = delta
    with np.load(f"m_{Lx}x{Ly}_{δ}.npz") as f:
        mx, my, mz = f["mx"], f["my"], f["mz"]
        ic(mx, my, mz)
        ic(mx.shape, my.shape, mz.shape)
    
    # Define a function for magnetization at coordinates.
    def σ(i):
        x, y = i[:-1]
        return mx[x, y] * σ1 + my[x, y] * σ2 + mz[x, y] * σ3
    
    # Perform Tc calculations.
    with open(f"defect_{Lx}x{Ly}_{δ}.dat", "a") as f:
        # Model parameters.
        t = 1.0
        μ = 0.5
        m = mag
        U = t

        # Construct the Hamiltonian.
        system = Hamiltonian(lattice)
        with system as (H, Δ, V):
            for i in lattice.sites():
                H[i, i] = -μ * σ0 - m * σ(i)
                V[i, i] = -U

            for i, j in lattice.bonds():
                H[i, j] = -t * σ0

        # Calculate the critical temperature.
        Tc = critical_temperature(system, T_max=0.01, bisects=10, iters=6)

        # Save the results to file.
        f.write(f"{Lx}x{Ly}, {δ}, {m}, {Tc}\n")
        f.flush()

if __name__ == "__main__":
    ic()
    run(main)
    ic()
