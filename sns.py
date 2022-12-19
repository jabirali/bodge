#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from bodge import *

# Define lengths of the materials in an SC/NM/AM/NM/SC system.
L_X = 64
L_Y = 11
L_Z = 1

lattice = CubicLattice((L_X,L_Y,L_Z))

# Parameters corresponding to the lattice model.
t = 1.0
Δ0 = t
δφ = π/2
T = 0.01*t

system = Hamiltonian(lattice)
with system as (H, Δ, V):
    for i in lattice.sites():
        if i[0] < 10:
            Δ[i, i] = -Δ0 * jσ2 * np.exp((-1j/2) * δφ)
        elif i[0] >= L_X - 10:
            Δ[i, i] = -Δ0 * jσ2 * np.exp((+1j/2) * δφ)
    for i, j in lattice.bonds():
        H[i, j] = -t * σ0

Ns = []
Js = []
for N in tqdm([100, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]):
    fermi = FermiMatrix(system, N)
    J = sum(fermi(T).current_elec(axis=0)[(L_X//2, y, L_Z//2)] for y in range(L_Y))

    Ns.append(N)
    Js.append(J)

    plt.figure()
    plt.plot(Ns, Js)
    plt.show()