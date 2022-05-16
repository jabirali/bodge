#!/usr/bin/env python

import numpy as np
from matplotlib.pyplot import legend, plot, show, xlabel, xlim, ylabel

from bodge import *

t = 1
Δ0 = t
μ = 0
m3 = 0

if __name__ == "__main__":
    lattice = CubicLattice((6, 6, 6))
    hamiltonian = Hamiltonian(lattice)
    solver = Solver(chebyshev, hamiltonian, blocksize=32, energies=512, resolve=True)
    solver2 = Solver(chebyshev, hamiltonian, blocksize=32, energies=512, radius=20, resolve=True)

    with hamiltonian as (H, Δ):
        for i in lattice.sites():
            H[i, i] = -μ * σ0 
            Δ[i, i] = Δ0 * jσ2

        for i, j in lattice.bonds():
            H[i, j] = -t * σ0

    sol = solver()
    sol2 = solver2()
    # print(sol.integral)

    ws = []
    D1 = []
    for ω, A in sol.spectral():
        # x = sol.lattice[1, 1, 1]
        dof = A.blocksize[0]
        A_ii = A.diagonal()
        A_up = np.mean( A_ii[0::dof] )
        A_dn = np.mean( A_ii[1::dof] )
        # print(len(A_ii), len(A_up), len(A_dn))
        D1.append(A_up + A_dn)
        ws.append(ω)
        # D = A[x]
        # print(ω, D)

    D3 = []
    for ω, A in sol2.spectral():
        dof = A.blocksize[0]
        A_ii = A.diagonal()
        A_up = np.mean( A_ii[0::dof] )
        A_dn = np.mean( A_ii[1::dof] )
        D3.append(A_up + A_dn)

    # ws = [w * hamiltonian.scale for w in ws]
    # D1 = [d1 / hamiltonian.scale for d1 in D1]

    D2 = []
    # print(hamiltonian.scale)
    A2 = hamiltonian.spectralize(ws, resolution=0.7e-2)
    for A in A2:
        A_ii = A.diagonal()
        A_up = np.mean( A_ii[0::dof] )
        A_dn = np.mean( A_ii[1::dof] )
        D2.append(A_up + A_dn)

    plot(ws, D2, "k", ws, D3, "b-", ws, D1, "r--")
    legend([ "Direct solution", "Chebyshev (no cutoff)", "Chebyshev (radius 5)"])
    xlabel(r"Energy $\omega/t$")
    ylabel("Density of states")
    xlim([-15, +15])
    show()

# for ω, A in sol.spectral:
# print(ω, A)
