#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from bodge import *

# Define lengths of the materials in an SC/NM/AM/NM/SC system.
L_SC = 10
L_NM = 3
L_AM = 10

L_X = 2 * L_SC + 2*L_NM + L_AM
L_Y = 11
L_Z = 1

lattice = CubicLattice((L_X,L_Y,L_Z))

# Function to handle diagonal slicing of junctions.
def diax(i, n):
    return i[0] + (i[1] - L_Y//2) * n

# Parameters corresponding to the lattice model.
t = 1.0
h = 0.15 * t
μ = - t
Δ0 = - t
T = 0.01*t

# Parameters used for exploration.
φs = np.linspace(0.0, 2.0 * π, 100)
Js = np.zeros_like(φs)

# Iterate over phase differences.
for n, δφ in enumerate( tqdm( φs ) ):
    # Create the relevant system.
    system = Hamiltonian(lattice)
    fermi = FermiMatrix(system, 1000)

    with system as (H, Δ, V):
        # On-site interactions.
        for i in lattice.sites():
            # Every site has same chemical potential.
            H[i, i] = -μ * σ0

            # Left superconductor.
            if diax(i, 1) < L_SC:
                Δ[i, i] = Δ0 * jσ2 * np.exp((-1j/2) * δφ)

            # Right superconductor.
            elif diax(i, 1) >= L_X - L_SC:
                Δ[i, i] = Δ0 * jσ2 * np.exp((+1j/2) * δφ)
        
        # Hopping along x-axis.
        # TODO: Check hopping b.c.
        for i, j in lattice.bonds(axis=0):
            # Inside altermagnet.
            if diax(i, 1) >= (L_SC + L_NM) and diax(i, 1) < L_X - (L_SC + L_NM):
                H[i, j] = -t * σ0 - h * σ3
            elif diax(j, 1) >= (L_SC + L_NM) and diax(j, 1) < L_X - (L_SC + L_NM):
                H[i, j] = -t * σ0 - h * σ3
            
            # Outside altermagnet.
            else:
                H[i, j] = -t * σ0

        # Hopping along y-axis.
        for i, j in lattice.bonds(axis=1):
            # Inside altermagnet.
            if diax(i, 1) >= (L_SC + L_NM) and diax(i, 1) < L_X - (L_SC + L_NM):
                H[i, j] = -t * σ0 + h * σ3
            elif diax(j, 1) >= (L_SC + L_NM) and diax(j, 1) < L_X - (L_SC + L_NM):
                H[i, j] = -t * σ0 + h * σ3
            
            # Outside altermagnet.
            else:
                H[i, j] = -t * σ0

    # Calculate the current.
    Js[n] = fermi(T).current_elec(axis=0)[(L_SC + L_NM//2, L_Y//2, L_Z//2)]

plt.plot(φs, Js)
plt.show()