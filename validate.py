#!/usr/bin/env python

from bodge import *
from h5py import File

t = 1.0
μ = +3 * t
Δ0 = t / 2
m3 = t / 5

if __name__ == "__main__":
    lattice = CubicLattice((16, 16, 16))
    hamiltonian = Hamiltonian(lattice)
    spectral = ChebyshevSolver(hamiltonian)

    with hamiltonian as (H, Δ):
        for i in lattice.sites():
            H[i, i] = -μ * σ0 - m3 * σ3
            Δ[i, i] = Δ0 * jσ2

        for i, j in lattice.bonds():
            H[i, j] = -t * σ0

    f = File(spectral(), 'r')
    print([k for k in f['0001']])

    # print(spectral.solution[4])
