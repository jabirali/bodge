#!/usr/bin/env python

import numpy as np
from matplotlib.pyplot import legend, plot, show, xlabel, xlim, ylabel

from bodge import *

t = 1
Δ0 = t / 2
μ = 0.25
m3 = 0

if __name__ == "__main__":
    lattice = CubicLattice((100, 1, 1))
    hamiltonian = Hamiltonian(lattice)
    solver = Solver(chebyshev, hamiltonian, blocksize=25, energies=512, resolve=True)
    # solver2 = Solver(chebyshev, hamiltonian, blocksize=32, energies=512, radius=20, resolve=True)

    with hamiltonian as (H, Δ):
        for i in lattice.sites():
            H[i, i] = -μ * σ0 - m3 * σ3
            Δ[i, i] = Δ0 * jσ2

        for i, j in lattice.bonds():
            H[i, j] = -t * σ0

    sol = solver()

    ws = []
    D0 = []
    D1 = []
    D2 = []
    x0 = sol.lattice[0, 1, 1]
    x1 = sol.lattice[1, 1, 1]
    x2 = sol.lattice[30, 1, 1]
    for ω, A in sol.spectral():
        dof = A.blocksize[0]
        A_ii = A.diagonal()
        dos = np.real(A_ii[0::dof] + A_ii[1::dof])
        D0.append(dos[x0])
        D1.append(dos[x1])
        D2.append(dos[x2])
        ws.append(ω)

    # A2 = hamiltonian.spectralize(ws, 0.1)
    # D2 = []
    # for A in A2:
    #     A_ii = A.diagonal()
    #     A_up = A_ii[0::dof]
    #     A_dn = A_ii[1::dof]
    #     D2.append(A_up[x] + A_dn[x])

    plot(ws, D0, "k", ws, D1, "b", ws, D2, "r")
    legend(["i = 0", "i = 1", "i = 30"])
    xlabel(r"Energy $\omega/t$")
    ylabel("Density of states")
    show()

    # plot(ws, D)

# for ω, A in sol.spectral:
# print(ω, A)
