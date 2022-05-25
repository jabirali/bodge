#!/usr/bin/env python

"""Validation script to test DOS calculations."""

import matplotlib.pyplot as plt
import numpy as np

from bodge import *

# Physical parameters.
t = 1
Δ0 = 0.5
μ = 0.5
m3 = 0.25

# Numerical parameters.
params = [(400, None)]

# Perform the validation.
if __name__ == "__main__":
    # Construct a 1D test system.
    lattice = CubicLattice((64, 1, 1))
    hamiltonian = Hamiltonian(lattice)
    with hamiltonian as (H, Δ):
        for i in lattice.sites():
            H[i, i] = -μ * σ0 - m3 * σ3
            Δ[i, i] = Δ0 * jσ2

        for i, j in lattice.bonds():
            H[i, j] = -t * σ0

    # Perform simulations.
    x = lattice[31, 0, 0]
    results = {}
    for energy, radius in params:
        # Instantiate solver.
        solver = Solver(
            chebyshev,
            hamiltonian,
            blocksize=8,
            energies=energy,
            radius=radius,
            resolve=True,
        )

        # Calculate density of states.
        results[energy, radius] = solver().density()

    # Plot the results.
    fig, ax = plt.subplots()
    for (energy, radius), (ω, ρ) in reversed(results.items()):
        ρ = ρ[x, :]
        print(energy, radius, ρ[np.argmin(np.abs(ω))])
        ax.plot(ω[:], ρ[:], label=f"E={energy}, R={solver.radius} radius")
    # plt.xlim([-4, 4])
    # plt.ylim([0, 1])
    plt.xlabel(r"Energy $\omega/t$")
    plt.ylabel(r"Density of states $\rho(\omega)$")
    # plt.legend()
    plt.show()
