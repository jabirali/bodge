#!/usr/bin/env python

"""Revised exploration of superconductor-altermagnet junctions.

Based on some initial promising results in `altermagnets.ipynb`, we now attempt
to further explore the physics of 0-π oscillations in Josephson junctions with
altermagnetic interlayers -- using e.g. more realistic material parameters.

This time around, I'm using Jupytext notation instead of Jupyter notebooks,
so that roughly the same scripts can more easily be run on an HPC facility.
"""

# %% Common imports.
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm, trange

from bodge import *

# %% Lattice geometry.
DIAG: bool  # Diagonal [110] vs. straight [100] junctions

L_SC: int  # Superconductor length
L_NM: int  # Normal-metal length
L_AM: int  # Altermagnet length

L_X: int  # Total length along the junction direction [X]
L_Y: int  # Total length in the transverse direction [Y]
L_Z: int = 1  # We consider only 2D lattices here

# %% Lattice coordinates.
# These functions transform lattice coordinates i = (i_x, i_y, 0) to
# junction coordinates x_i and y_i along either [100] or [110] systems.
def x(i):
    """Junction coordinate along the junction axis."""
    if not DIAG:
        return i[0]
    else:
        return (i[0] + (L_X - i[1]))//2

def y(i):
    """Junction coordinate in the transverse direction."""
    if not DIAG:
        return i[1]
    else:
        return (i[0] - (L_X - i[1]))//2

# Check if a coordinate i is within the junction or not. Useful
# for diagonal systems where we need a larger lattice than junction.
def inside(i):
    return x(i) >= 0 and x(i) < L_X and y(i) >= 0 and y(i) < L_Y

# %% Material coordinates.
# These functions are used to check whether a specific lattice site i
# corresponds to a specific material. Nice for Hamiltonian construction.
def SC1(i):
    """Left superconductor."""
    return x(i) < L_SC
    
def SC2(i):
    """Right superconductor."""
    return x(i) >= L_X - L_SC

def NM1(i):
    """Left normal spacer."""
    return x(i) < L_SC + L_NM and not SC1(i)

def NM2(i):
    """Right normal spacer."""
    return x(i) >= L_X - L_SC - L_NM and not SC2(i)

def OBS(i):
    """Current observation region."""
    return x(i) == L_SC + L_NM//2

def AM(i):
    """Altermagnetic interlayer."""
    return not SC1(i) and not SC2(i) and not NM1(i) and not NM2(i)

# %% Lattice construction.
# These routines are used to construct and visualize the relevant lattice types.
lattice: CubicLattice
def create_lattice():
    if DIAG:
        return CubicLattice((L_X + L_Y, L_X + L_Y, 1))
    else:
        return CubicLattice((L_X, L_Y, 1))

def visualize():
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.set_axis_off()
    marker="."

    for i in lattice.sites():
        if not inside(i):
            ax.scatter(x=i[0], y=i[1], color='#eeeeee', marker=marker)
        elif OBS(i):
            ax.scatter(x=i[0], y=i[1], color='#ff0000', marker=marker)
        elif SC1(i) or SC2(i):
            ax.scatter(x=i[0], y=i[1], color='#ff7f00', marker=marker)
        elif NM1(i) or NM2(i):
            ax.scatter(x=i[0], y=i[1], color='k', marker=marker)
        elif AM(i):
            ax.scatter(x=i[0], y=i[1], color='#984ea3', marker=marker)
    plt.show()

# %% Simple lattice test.
L_NM = 3
L_AM = 32

DIAG = False

L_SC = 11
L_Y = 11
L_X = 2 * L_SC + 2 * L_NM + L_AM

lattice = create_lattice()
visualize()

DIAG = True

L_SC = 8
L_Y = 8
L_X = 2 * L_SC + 2 * L_NM + L_AM

lattice = create_lattice()
visualize()

# %% Current calculations.
def current(system, N=1000, T=1e-3):
    F = FermiMatrix(system, N)(T)
    Jx = F.current_elec(axis=0)
    Jy = F.current_elec(axis=1)

    J = 0.0
    for i in lattice.sites():
        if OBS(i):
            if DIAG:
                # Current in 45º direction.
                J += (Jx[i] - Jy[i])/np.sqrt(2)
            else:
                # Current in 0º direction.
                J += Jx[i]

    return J

# %% Check current convergence for S/F/S junctions.
t = 1.0
Δ0 = 0.1 * t
μ = -0.5 * t
δφ = π/2

m = Δ0/2

system = Hamiltonian(lattice)
with system as (H, Δ, V):
    for i in lattice.sites():
        if inside(i):
            if AM(i):
                H[i, i] = -μ * σ0 - m * σ3
            else:
                H[i, i] = -μ * σ0

            if SC1(i):
                Δ[i, i] = Δ0 * jσ2 * np.exp((-1j/2) * δφ)
            if SC2(i):
                Δ[i, i] = Δ0 * jσ2 * np.exp((+1j/2) * δφ)
    for i, j in lattice.bonds():
        if inside(i) and inside(j):
            H[i, j] = -t * σ0
        
Ns = []
Js = []
for N in tqdm([200, 400, 800, 1600, 2000, 2400, 2800, 3200, 3600, 4000]):
    Ns.append(N)
    Js.append(current(system, N))
    print(f"J(N = {Ns[-1]})/t = {Js[-1]}")

plt.plot(Ns, Js)
plt.xlabel(r"Chebyshev order $N$")
plt.ylabel(r"Supercurrent $J(π/2)/t$")

# %% Similar test for smaller gaps.
t = 1.0
Δ0 = 0.01 * t
μ = -0.5 * t
δφ = π/2

Tc = (Δ0 / 1.764)
T = 0.1 * Tc

m = Δ0/2

system = Hamiltonian(lattice)
with system as (H, Δ, V):
    for i in lattice.sites():
        if inside(i):
            if AM(i):
                H[i, i] = -μ * σ0 - m * σ3
            else:
                H[i, i] = -μ * σ0

            if SC1(i):
                Δ[i, i] = Δ0 * jσ2 * np.exp((-1j/2) * δφ)
            if SC2(i):
                Δ[i, i] = Δ0 * jσ2 * np.exp((+1j/2) * δφ)
    for i, j in lattice.bonds():
        if inside(i) and inside(j):
            H[i, j] = -t * σ0
        
Ns = []
Js = []
for N in tqdm([200, 400, 800, 1600, 2000, 2400, 2800, 3200, 3600, 4000]):
    Ns.append(N)
    Js.append(current(system, N, T))
    print(f"J(N = {Ns[-1]})/t = {Js[-1]}")

plt.plot(Ns, Js)
plt.xlabel(r"Chebyshev order $N$")
plt.ylabel(r"Supercurrent $J(π/2)/t$")
# %%
