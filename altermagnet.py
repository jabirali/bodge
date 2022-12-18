#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from bodge import *

# Define lengths of the materials in an SC/NM/AM/NM/SC system.
L_SC = 20
L_NM = 10
L_AM = 20

angle = 45 # or 0

Lx = 2 * L_SC + 2 * L_NM + L_AM
Ly = 1
Lz = 1


lattice = CubicLattice((Lx, Ly, Lz))
system = Hamiltonian(lattice)
fermi = FermiMatrix(system, 1000)

t = 1.0
h = 0.15 * t
μ = - t
Δ0 = - t
T = 0.01*t
 
φs = np.linspace(0.0, 2.0, 100)
Js = np.zeros_like(φs)

# Instantiate the system.
for n, δφ in enumerate( tqdm( φs ) ):
    with system as (H, Δ, V):
        for i in lattice.sites():
            H[i, i] = -μ * σ0
            if i[0] < L_sc:
                Δ[i, i] = Δ0 * jσ2 * np.exp((-1j/2) * π * δφ)
            if i[0] >= Lx - L_sc:
                Δ[i, i] = Δ0 * jσ2 * np.exp((+1j/2) * π * δφ)

        for i, j in lattice.bonds(axis=0):
            H[i, j] = -t * σ0 - h * σ3
        for i, j in lattice.bonds(axis=1):
            H[i, j] = -t * σ0 + h * σ3

    # Calculate the current.
    Js[n] = fermi(T).current_elec(axis=0)[(Lx//2, Ly//2, Lz//2)]

plt.plot(φs, Js)
plt.show()