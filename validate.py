#!/usr/bin/env python

import scipy.sparse as sp

from bodge import *

t = 1.0
μ = +3 * t
Δ0 = t / 2
m3 = t / 5

if __name__ == "__main__":
    lattice = CubicLattice((8, 8, 8))
    hamiltonian = Hamiltonian(lattice)
    solver = ChebyshevSolver(hamiltonian)

    with hamiltonian as (H, Δ):
        for i in lattice.sites():
            H[i, i] = -μ * σ0 - m3 * σ3
            Δ[i, i] = Δ0 * jσ2

        for i, j in lattice.bonds():
            H[i, j] = -t * σ0

    for A_m, ω_m, w_m in solver():
        # print(A_m)
        print(ω_m)
        # print(w_m)
        # print("=====")
