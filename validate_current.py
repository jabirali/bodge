#!/usr/bin/env python

"""Validation script to test current calculations."""

import numpy as np

from bodge import *

# Physical parameters.
t = 1
Δ0 = 1
μ = 0.5
m3 = 0.5
ds = reversed([2*i for i in range(1, 16)])

# Numerical parameters.
#params = [ (200 * n, 64) for n in range(1, 12) ]
params = [(1000, 64)]
# phases = np.linspace(0, 2*π, 1+8*2)
phases = [π/2]

# Perform the validation.
if __name__ == "__main__":
    # Construct a 1D test system.
    lattice = CubicLattice((128, 1, 1))
    hamiltonian = Hamiltonian(lattice)
    with hamiltonian as (H, Δ):
        for i, j in lattice.bonds():
            H[i, j] = -t * σ0

    # Define interlayer.
    for d in ds:
        x1 = (lattice.shape[0] - d) // 2
        x2 = (lattice.shape[0] + d) // 2

        # Perform simulations.
        Ep = 0
        Jp = 0
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
                        if i[0] < x1:
                            H[i, i] = -μ * σ0
                            Δ[i, i] = (Δ0 * np.exp(-1j * φ / 2)) * jσ2
                        elif i[0] > x2:
                            H[i, i] = -μ * σ0
                            Δ[i, i] = (Δ0 * np.exp(+1j * φ / 2)) * jσ2
                        else:
                            H[i, i] = -μ * σ0 -m3 * σ3

                results = solver().integral()

                J = {}
                mid = 0
                for i, j in lattice.bonds(axis=0):
                    k = hamiltonian.index(i, j)
                    # - for electron, - included in t
                    Js = (
                        2
                        * hamiltonian.scale
                        * np.imag(
                            np.trace(hamiltonian.data[k, 0:2, 0:2] * results.data[k, 0:2, 0:2])
                        )
                    )
                    if i[0] > x1 and i[0] < x2:
                        try:
                            J[i[0]] += Js
                        except:
                            J[i[0]] = Js
                        if i[0] == (x1+x2)//2:
                            mid += Js

                J = np.array([Js for Js in J.values()])
                diff = np.abs(J.max() - J.min())
                mean = np.abs(np.mean(J))
                tot = np.sum(J)
                print(d, energy, radius, φ, mid, diff, np.abs(tot - Jp)/(np.abs(tot) + np.abs(Jp)), tot)
                Jp = tot

                # print(energy, φ, J)
