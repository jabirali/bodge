#!/usr/bin/env python

"""Josephson junctions with altermagnetic interlayers."""

# %% Common imports.
import numpy as np
from argparse import ArgumentParser

from bodge import *


# %% Basic function definitions.
def X(i):
    """Junction coordinate along the junction axis."""
    return i[0]

def Y(i):
    """Junction coordinate in the transverse direction."""
    return i[1]

def INS(i):
    """Check if a coordinate is inside the junction."""
    return X(i) < W or Y(i) < W

def SC1(i):
    """First superconductor."""
    return Y(i) >= W + L_A1 + L_NM
  
def SC2(i):
    """Second superconductor."""
    return X(i) >= W + L_A2 + L_NM

def NM1(i):
    """First normal spacer."""
    return Y(i) >= W + L_A1 and not SC1(i)

def NM2(i):
    """Second normal spacer."""
    return X(i) >= W + L_A2 and not SC2(i)

def OBS1(i):
    """Current observation region."""
    return Y(i) == W + L_A1 + 1

def OBS2(i):
    """Current observation region."""
    return X(i) == W + L_A2 + 1

def AM(i):
    """Altermagnetic interlayer."""
    return not SC1(i) and not SC2(i) and not NM1(i) and not NM2(i)

def current(system, N=2500, T=0.0):
    F = FermiMatrix(system, N)(T)
    Jx = F.current_elec(axis=0)
    Jy = F.current_elec(axis=1)

    J1 = 0.0
    J2 = 0.0
    for i in lattice.sites():
        if OBS1(i):
            J1 += Jx[i]
        if OBS2(i):
            J2 += Jy[i]

    return J1, J2

def visualize():
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.set_axis_off()
    marker="."

    for i in lattice.sites():
        if INS(i):
            if OBS1(i) or OBS2(i):
                ax.scatter(x=i[0], y=i[1], color='#ff0000', marker=marker)
            elif SC1(i) or SC2(i):
                ax.scatter(x=i[0], y=i[1], color='#ff7f00', marker=marker)
            elif NM1(i) or NM2(i):
                ax.scatter(x=i[0], y=i[1], color='k', marker=marker)
            elif AM(i):
                ax.scatter(x=i[0], y=i[1], color='#984ea3', marker=marker)
            else:
                ax.scatter(x=i[0], y=i[1], color='#eeeeee', marker=marker)
    plt.show()

# %% Prepare lattice.
W = 20

L_SC = W
L_NM = 3
L_A1 = 2
L_A2 = 10

L_Y = L_SC + L_NM + L_A1 + W
L_X = L_SC + L_NM + L_A2 + W
L_Z = 1

# Construct and visualize.
lattice = CubicLattice((L_X, L_Y, 1))
visualize()

# %% Tight-binding parameters.
t = 1.0
Δ0 = 0.1 * t
μ = -0.5 * t
m = 1.5 * Δ0

Tc = (Δ0 / 1.764)
T = 0.1 * Tc

# %% Perform the calculations
δφ = 0.5

system = Hamiltonian(lattice)
with system as (H, Δ, V):
    for i in lattice.sites():
        if INS(i):
            H[i, i] = -μ * σ0

            if SC1(i):
                Δ[i, i] = Δ0 * jσ2 * np.exp((-1j/2) * π * δφ)
            if SC2(i):
                Δ[i, i] = Δ0 * jσ2 * np.exp((+1j/2) * π * δφ)
    for i, j in lattice.bonds(axis=0):
        if INS(i) and INS(j):
            if AM(i) and AM(j):
                H[i, j] = -t * σ0 - m * σ3
            else:
                H[i, j] = -t * σ0
    for i, j in lattice.bonds(axis=1):
        if INS(i) and INS(j):
            if AM(i) and AM(j):
                H[i, j] = -t * σ0 + m * σ3
            else:
                H[i, j] = -t * σ0

J1, J2 = current(system, T=T)
print(J1, J2)

# %%
