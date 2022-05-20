#!/usr/bin/env python

import numpy as np
from matplotlib.pyplot import axes, legend, plot, show, subplot, subplots, xlabel, xlim, ylabel

from bodge import *

t = 1
Δ0 = 0
μ = 0.5
m3 = 0

if __name__ == "__main__":
    lattice = CubicLattice((10, 10, 10))
    hamiltonian = Hamiltonian(lattice)
    with hamiltonian as (H, Δ):
        for i in lattice.sites():
            H[i, i] = -μ * σ0 - m3 * σ3
            Δ[i, i] = Δ0 * jσ2

        for i, j in lattice.bonds():
            H[i, j] = -t * σ0

    x = lattice[5,5,5]
    ws = []
    Ds = []
    radii = [4]
    for radius in radii:
        w = []
        D = []
        solver = Solver(chebyshev, hamiltonian, blocksize=200, energies=200, resolve=True, radius=radius)
        sol = solver()
        for ω, A in sol.spectral():
            dof = A.blocksize[0]
            A_ii = A.diagonal()
            dos = np.real(A_ii[0::dof] + A_ii[1::dof])

            w.append(ω)
            D.append(dos[x])
        Ds.append(np.array(D))
        ws.append(w)


    # A2 = hamiltonian.spectralize(ws, 0.1)
    # D2 = []
    # for A in A2:
    #     A_ii = A.diagonal()
    #     A_up = A_ii[0::dof]
    #     A_dn = A_ii[1::dof]
    #     D2.append(A_up[x] + A_dn[x])

    fig, ax = subplots()
    for w, D in zip(ws, Ds):
        ax.plot(w, D)
    legend(radii)
    # legend(["i = 0", "i = 1", "i = 30"])
    xlabel(r"Energy $\omega/t$")
    ylabel("Density of states")
    show()

    # plot(ws, D)

# for ω, A in sol.spectral:
# print(ω, A)
