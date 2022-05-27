#!/usr/bin/env python

"""Validation script to test DOS calculations."""

import matplotlib.pyplot as plt
import numpy as np

from bodge import *

# Physical parameters.
t = 1
μ = 1
Lx = 16
Ly = 16
Δ0 = 0.01

# Perform the validation.
if __name__ == "__main__":
    # Construct a 1D test system.
    lattice = CubicLattice((Lx, Ly, 1))
    hamiltonian = Hamiltonian(lattice)
    with hamiltonian as (H, Δ):
        for i in lattice.sites():
            H[i, i] = -μ * σ0
            Δ[i, i] = Δ0 * jσ2

        for i, j in lattice.bonds():
            H[i, j] = -t * σ0

    # Perform simulations.
    x = lattice[Lx // 2, Ly // 2, 0]
    ω = np.linspace(-2.5 * Δ0, +2.5 * Δ0, 64)
    ρ = np.zeros_like(ω)
    η = 0.1 * Δ0

    As = hamiltonian.spectralize(ω, η)
    for m, A in enumerate(As):
        A_ii = A.diagonal()
        dos = np.real(A_ii[0::4] + A_ii[1::4])
        ρ[m] = dos[x]

    # Plot the results.
    fig, ax = plt.subplots()
    ax.plot(ω[:], ρ[:], label=rf"\eta = {η:.2f}Δ")
    plt.xlabel(r"Energy $\omega/t$")
    plt.ylabel(r"Exact density of states $\rho(\omega)$")
    # plt.legend()
    plt.show()
