#!/usr/bin/env python

"""Validation script to test DOS calculations."""

import matplotlib.pyplot as plt
import numpy as np

from bodge import *

# Physical parameters.
t = 1
μ = 0.5
Lx = 512
Ly = 2
Lz = 1
Δ0 = 0.04

# Numerical parameters.
params = [(1200, Lx // 2)]

# Perform the validation.
if __name__ == "__main__":
    # Construct a 1D test system.
    lattice = CubicLattice((Lx, Ly, Lz))
    hamiltonian = Hamiltonian(lattice)
    with hamiltonian as (H, Δ):
        for i in lattice.sites():
            H[i, i] = -μ * σ0
            Δ[i, i] = Δ0 * jσ2

        for i, j in lattice.bonds():
            H[i, j] = -t * σ0

    # Perform simulations.
    x = lattice[Lx // 2, Ly // 2, Lz // 2]
    results = {}
    for energy, radius in params:
        # Instantiate solver.
        solver = Solver(
            ChebyshevKernel,
            hamiltonian,
            blocksize=1024 // 16,
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
        ax.plot(ω[:], ρ[:], label=f"E={energy}, R={radius}")
    # plt.xlim([-5*Δ0, 5*Δ0])
    # plt.ylim([0, 1])
    plt.xlabel(r"Energy $\omega/t$")
    plt.ylabel(r"Density of states $\rho(\omega)$")
    plt.legend()
    plt.show()
