#!/usr/bin/env python

import scipy.sparse as sp

from bodge import *
from bodge.solver2 import BlockSolver, Solver

t = 1.0
μ = +3 * t
Δ0 = t / 2
m3 = t / 5

if __name__ == "__main__":
    lattice = CubicLattice((96, 96, 1))
    hamiltonian = Hamiltonian(lattice)
    # solver = ChebyshevSolver(hamiltonian)
    solver = Solver(hamiltonian, BlockSolver)

    with hamiltonian as (H, Δ):
        for i in lattice.sites():
            H[i, i] = -μ * σ0 - m3 * σ3
            Δ[i, i] = Δ0 * jσ2

        for i, j in lattice.bonds():
            H[i, j] = -t * σ0

    solver()
    # for A_m, ω_m, w_m in solver():
    # print(A_m)
    # print(ω_m)
    # print(w_m)
    # print("=====")
