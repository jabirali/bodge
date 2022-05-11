#!/usr/bin/env python


from bodge import *

t = 1.0
μ = +3 * t
Δ0 = t / 2
m3 = t / 5

if __name__ == "__main__":
    lattice = CubicLattice((32, 32, 10))
    hamiltonian = Hamiltonian(lattice)
    solver = Solver(chebyshev, hamiltonian, resolve=False)

    with hamiltonian as (H, Δ):
        for i in lattice.sites():
            H[i, i] = -μ * σ0 - m3 * σ3
            Δ[i, i] = Δ0 * jσ2

        for i, j in lattice.bonds():
            H[i, j] = -t * σ0

    sol = solver()
    print(sol.integral)

    # for ω, A in sol.spectral:
    # print(ω, A)
