"""Simple test script for self-consistency iterations."""

import numpy as np
from numpy.random import rand

from bodge import *
from bodge.utils import singlet

lattice = CubicLattice((1000, 1, 1))
system = Hamiltonian(lattice)

Δ0 = (0.01 + 0.01j) * np.ones(lattice.shape)
α = 0.1

for n in range(1000):
    with system as (H, Δ):
        for i in lattice.sites():
            Δ[i, i] = Δ0[*i] * jσ2
        for i, j in lattice.bonds():
            H[i, j] = -1 * σ0

    eigs = system.diagonalize(format="wave", cuda=True)

    new = singlet(eigs, 3.0)
    diff = np.mean(np.abs(new - Δ0))
    if n < 10:
        # Fast initial convergence
        Δ0 = 2 * new - Δ0
    else:
        # Slow later convergence
        Δ0 = α * new + (1-α) * Δ0

    g = np.mean(Δ0)
    print(np.abs(g), np.angle(g, deg=True), diff)
