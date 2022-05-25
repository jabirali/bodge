#!/usr/bin/env python

"""Validation script to test current calculations."""

import numpy as np

from bodge import *

# Physical parameters.
t = 1
Δ0 = 1
μ = 0.5
m3 = 0.0
d = 16

# Numerical parameters.
params = [
    (200, None),
    (400, None),
    (600, None),
    (800, None),
    (1000, None),
    (1200, None),
    (1400, None),
    (1600, None),
    (1800, None),
    (2000, None),
]
phases = [π / 2]

# Perform the validation.
if __name__ == "__main__":
    # Construct a 1D test system.
    lattice = CubicLattice((48, 16, 1))
    hamiltonian = Hamiltonian(lattice)
    with hamiltonian as (H, Δ):
        for i in lattice.sites():
            H[i, i] = -μ * σ0 - m3 * σ3

        for i, j in lattice.bonds():
            H[i, j] = -t * σ0

    # Define interlayer.
    x1 = (lattice.shape[0] - d) // 2
    x2 = (lattice.shape[0] + d) // 2

    # Perform simulations.
    for energy, radius in params:
        # Instantiate solver.
        solver = Solver(
            chebyshev,
            hamiltonian,
            blocksize=lattice.shape[0] // 3,
            energies=energy,
            radius=radius,
            resolve=False,
        )

        # Calculate density of states.
        for φ in phases:
            with hamiltonian as (H, Δ):
                for i in lattice.sites():
                    if i[0] < x1:
                        Δ[i, i] = (Δ0 * np.exp(-1j * φ / 2)) * jσ2
                    if i[0] > x2:
                        Δ[i, i] = (Δ0 * np.exp(+1j * φ / 2)) * jσ2

            results = solver().integral()

            J = {}
            for i, j in lattice.bonds(axis=0):
                k = hamiltonian.index(i, j)
                # - for electron, - included in t
                Js = (
                    2
                    * hamiltonian.scale
                    * np.imag(
                        +hamiltonian.data[k, 0, 0] * results.data[k, 0, 0]
                        + hamiltonian.data[k, 1, 1] * results.data[k, 1, 1]
                    )
                )
                if i[0] > x1 and i[0] < x2:
                    try:
                        J[i[0]] += Js
                    except:
                        J[i[0]] = Js

            J = np.array([Js for Js in J.values()])
            diff = np.abs(J.max() - J.min())
            mean = np.abs(np.mean(J))
            tot = np.sum(J)
            print(energy, φ, diff / (mean + 1e-100))

            # print(energy, φ, J)
