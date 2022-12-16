#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from bodge import *

Lx = 32
Ly = 1
Lz = 1

Ls = 20

lattice = CubicLattice((Lx, Ly, Lz))
system = Hamiltonian(lattice)
fermi = FermiMatrix(system, 1000)

t = 1.0
h = 0.0
μ = - t
Δ0 = - t
T = 0.01*t
 
φs = np.linspace(0.0, 2.0, 100)
Js = np.zeros_like(φs)

# Instantiate the system.
for n, δφ in enumerate( tqdm( φs ) ):
    with system as (H, Δ, V):
        for i in lattice.sites():
            # Normal contribution.
            H[i, i] = -μ * σ0

            # Superconductors.
            # TODO: Complex t_ij!
            if i[0] < Ls:
                Δ[i, i] = Δ0 * jσ2 * np.exp((-1j/2) * π * δφ)
            if i[0] >= Lx - Ls:
                Δ[i, i] = Δ0 * jσ2 * np.exp((+1j/2) * π * δφ)

        for i, j in lattice.bonds(axis=0):
            H[i, j] = -t * σ0 - h * σ3
        for i, j in lattice.bonds(axis=1):
            H[i, j] = -t * σ0 + h * σ3

    # Calculate the current.
    Js[n] = fermi(T).current_elec(axis=0)[(Lx//2, Ly//2, Lz//2)]

plt.plot(φs, Js)
plt.show()