#!/usr/bin/env python

import sys
from bodge import *

t = 1.0
μ = +3 * t
Δ0 = t / 2
m3 = t / 5

if __name__ == "__main__":
    lattice = Cube((32, 8, 8))
    system = Hamiltonian(lattice)
    solver = Chebyshev(system)

    with system as (H, Δ):
        for i in lattice.sites():
            H[i, i] = -μ * σ0 - m3 * σ3
            Δ[i, i] = Δ0 * jσ2

        for i, j in lattice.neighbors():
            H[i, j] = -t * σ0

    result = solver.run()

    # TODO: Fix so that `solver.run()` uses up-to-date Hamiltonian.
    # G = solver.run(7)

    print(A[-0.9999922893814706])