#!/usr/bin/env python

"""Validation script to test DOS calculations."""

import matplotlib.pyplot as plt
from time import time
import numpy as np

from bodge import *

# Physical parameters.
t = 1
Δ0 = 0.5
μ = 0.5
m3 = 0.25

# Numerical parameters.
params = [(128, 8), (256, 16), (512, 32)]

# Perform the validation.
if __name__ == "__main__":
    # Construct a 1D test system.
    lattice = CubicLattice((256, 1, 1))
    hamiltonian = Hamiltonian(lattice)
    with hamiltonian as (H, Δ):
        for i in lattice.sites():
            H[i, i] = -μ * σ0 - m3 * σ3
            Δ[i, i] = Δ0 * jσ2

        for i, j in lattice.bonds():
            H[i, j] = -t * σ0

    # Perform simulations.
    x = lattice[64, 0, 0]
    for energy, radius in params:
        # Instantiate solver.
        solver = Solver(
            chebyshev,
            hamiltonian,
            blocksize=lattice.shape[0] // 8,
            energies=energy,
            radius=radius,
            resolve=True,
        )

        # Calculate density of states.
        t = time()
        ω, ρ = solver().density()
        t = time() - t

        # Write results to file.
        with open("validate_dos.dat", "a") as f:
            f.write(f"# E = {energy}, R = {radius}, T = {t}\n")
            for m in range(energy):
                f.write(f"{ω[m]},{ρ[x,m]}\n")
