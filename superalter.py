#!/usr/bin/env python

"""Josephson junctions with altermagnetic interlayers."""

import numpy as np
from argparse import ArgumentParser

from bodge import *


# ------------------------------------------------------------
# Part I: Basic function definitions.
# ------------------------------------------------------------

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

def inside(i):
    """Check if a coordinate is inside the junction."""
    return x(i) >= 0 and x(i) < L_X and y(i) >= 0 and y(i) < L_Y

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

def current(system, N=2500, T=0.0):
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


# ------------------------------------------------------------
# PART II: Prepare lattice and parameters.
# ------------------------------------------------------------

# Command-line arguments.
parser = ArgumentParser(description="Josephson junctions with altermagnetic interlayers")
parser.add_argument("-d", "--diagonal", default=False, action='store_true')
parser.add_argument("-l", "--length", type=int, required=True)
parser.add_argument("-m", "--field", type=float, required=True)
args = parser.parse_args()
print(args)

# Lattice parameters.
DIAG = bool(args.diagonal)
if DIAG:
    L_SC = 14
else:
    L_SC = 20

L_NM = 3
L_AM = args.length

L_X = 2 * L_SC + 2 * L_NM + L_AM
L_Y = 20
L_Z = 1

# Tight-binding parameters.
t = 1.0
Δ0 = 0.05 * t
μ = -0.5 * t
m = Δ0 * args.field

Tc = (Δ0 / 1.764)
T = 0.1 * Tc

# Lattice construction.
if DIAG:
    lattice = CubicLattice((L_X + L_Y, L_X + L_Y, 1))
else:
    lattice = CubicLattice((L_X, L_Y, 1))


# ------------------------------------------------------------
# Part III: Perform the actual calculations.
# ------------------------------------------------------------

φs = np.linspace(0.0, 2.0, 41)
for δφ in φs:
    system = Hamiltonian(lattice)
    with system as (H, Δ, V):
        for i in lattice.sites():
            if inside(i):
                H[i, i] = -μ * σ0

                if SC1(i):
                    Δ[i, i] = Δ0 * jσ2 * np.exp((-1j/2) * π * δφ)
                if SC2(i):
                    Δ[i, i] = Δ0 * jσ2 * np.exp((+1j/2) * π * δφ)
        for i, j in lattice.bonds(axis=0):
            if inside(i) and inside(j):
                if AM(i) and AM(j):
                    H[i, j] = -t * σ0 - m * σ3
                else:
                    H[i, j] = -t * σ0
        for i, j in lattice.bonds(axis=1):
            if inside(i) and inside(j):
                if AM(i) and AM(j):
                    H[i, j] = -t * σ0 + m * σ3
                else:
                    H[i, j] = -t * σ0

    J = current(system, T=T)
    print(f":: {DIAG},{L_SC},{L_NM},{L_AM},{m},{δφ},{J}")
