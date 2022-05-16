#!/usr/bin/env python

import numpy as np
from matplotlib.pyplot import legend, plot, show, xlabel, xlim, ylabel

from bodge import *

t = 1
Δ0 = t / 3
μ = t / 2
m3 = Δ0 / 2

if __name__ == "__main__":
    lattice = CubicLattice((4, 4, 4))
    hamiltonian = Hamiltonian(lattice)
    solver = Solver(chebyshev, hamiltonian, blocksize=32, energies=512, resolve=True)

    with hamiltonian as (H, Δ):
        for i in lattice.sites():
            H[i, i] = -μ * σ0 - m3 * σ3
            Δ[i, i] = Δ0 * jσ2

        for i, j in lattice.bonds():
            H[i, j] = -t * σ0

    sol = solver()
    # print(sol.integral)

    ws = []
    D1 = []
    x = lattice[1, 1, 1]
    for ω, A in sol.spectral():
        # x = sol.lattice[1, 1, 1]
        dof = A.blocksize[0]
        A_ii = A.diagonal()
        A_up = A_ii[0 + x * dof]
        A_dn = A_ii[1 + x * dof]
        # print(len(A_ii), len(A_up), len(A_dn))
        D1.append(A_up)
        ws.append(ω)
        # D = A[x]
        # print(ω, D)

    ws = [w * hamiltonian.scale for w in ws]
    D1 = [d1 / hamiltonian.scale for d1 in D1]

    D2 = []
    # print(hamiltonian.scale)
    A2 = hamiltonian.spectralize(ws, resolution=8e-3)
    for A in A2:
        A_ii = A.diagonal()
        A_ii = A_ii
        A_up = A_ii[0 + x * dof]
        A_dn = A_ii[1 + x * dof]
        D2.append(A_up)

    plot(ws, D1, "b", ws, D2, "r")
    legend([ "Chebyshev (no cutoff)", "Direct solution" ])
    xlabel(r"Energy $\omega/t$")
    ylabel("Density of states")
    xlim([-15, +15])
    show()

# for ω, A in sol.spectral:
# print(ω, A)
