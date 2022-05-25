#!/usr/bin/env python

"""Validation script to test current calculations."""

import matplotlib.pyplot as plt
import numpy as np

from bodge import *

# Physical parameters.
t = 1
Δ0 = 1
μ = 0.5
m3 = 0.0
d = 32

# Numerical parameters.
params = [(200, None), (400, None), (800, None), (1200, None)]
phases = [π / 2]

# Perform the validation.
if __name__ == "__main__":
    # Construct a 1D test system.
    lattice = CubicLattice((128, 1, 1))
    hamiltonian = Hamiltonian(lattice)
    with hamiltonian as (H, Δ):
        for i in lattice.sites():
            H[i, i] = -μ * σ0 - m3 * σ3

        for i, j in lattice.bonds():
            H[i, j] = -t * σ0

    # Perform simulations.
    x = (lattice.shape[0] // 2, lattice.shape[1] // 2, 0)
    for energy, radius in params:
        # Instantiate solver.
        solver = Solver(
            chebyshev,
            hamiltonian,
            blocksize=lattice.shape[0] // 8,
            energies=energy,
            radius=radius,
            resolve=False,
        )

        # Calculate density of states.
        for φ in phases:
            with hamiltonian as (H, Δ):
                for i in lattice.sites():
                    if i[0] < lattice.shape[0] // 2 - d // 2:
                        Δ[i, i] = (Δ0 * np.exp(-1j * φ / 2)) * jσ2
                    if i[0] > lattice.shape[0] // 2 + d // 2:
                        Δ[i, i] = (Δ0 * np.exp(+1j * φ / 2)) * jσ2

            results = solver().integral()

            J = 0
            for i, j in lattice.bonds(axis=0):
                if i == x:
                    k = hamiltonian.index(i, j)
                    # - for electron, - included in t
                    J += (
                        2
                        * hamiltonian.scale
                        * np.imag(
                            +hamiltonian.data[k, 0, 0] * results.data[k, 0, 0]
                            + hamiltonian.data[k, 1, 1] * results.data[k, 1, 1]
                        )
                    )

            print(energy, φ, J)
