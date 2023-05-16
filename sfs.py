#!/usr/bin/env python

"""Josephson junctions with altermagnetic interlayers."""

from argparse import ArgumentParser

# %% Common imports.
import numpy as np
from tqdm import tqdm

from bodge import *


# %% Check for interactivity.
def interactive():
    """Interactive session type. Returns `None` if run as a script."""
    try:
        return get_ipython().__class__.__name__
    except:
        return None


# %% Function definitions.
def x(i):
    """Junction coordinate along the junction axis."""
    if not DIAG:
        return i[0]
    else:
        return (i[0] + (L_X - i[1])) // 2


def y(i):
    """Junction coordinate in the transverse direction."""
    if not DIAG:
        return i[1]
    else:
        return (i[0] - (L_X - i[1])) // 2


def IN(i):
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


def OBS1(i):
    """Current observation region."""
    return x(i) == L_SC + 1


def OBS2(i):
    """Current observation region."""
    return x(i) == L_X - L_SC - 2


def AM(i):
    """Altermagnetic interlayer."""
    return not SC1(i) and not SC2(i) and not NM1(i) and not NM2(i)


def current(system, N=4000, T=0.0):
    F = FermiMatrix(system, N)(T)
    Jx = F.current_elec(axis=0)
    Jy = F.current_elec(axis=1)

    J1x = 0.0
    J2x = 0.0
    J1y = 0.0
    J2y = 0.0
    for i in lattice.sites():
        if OBS1(i):
            J1x += Jx[i]
            J1y += Jy[i]
        if OBS2(i):
            J2x += Jx[i]
            J2y += Jy[i]

    return J1x, J1y, J2x, J2y


def visualize():
    if args.visualize:
        import matplotlib.pyplot as plt

        win = "Bodge"
        fig = plt.figure(num=win, clear=True)

        fig, ax = plt.subplots(num=win)
        ax.set_aspect("equal")
        ax.set_axis_off()
        marker = "."

        NS = 0
        NN = 0
        NA = 0
        for i in lattice.sites():
            if IN(i):
                # if OBS1(i) or OBS2(i):
                #     ax.scatter(x=i[0], y=i[1], color="#ff0000", marker=marker)
                if SC1(i) or SC2(i):
                    NS += 1
                    ax.scatter(x=i[0], y=i[1], color="#ff7f00", marker=marker)
                elif NM1(i) or NM2(i):
                    NN += 1
                    ax.scatter(x=i[0], y=i[1], color="k", marker=marker)
                elif AM(i):
                    NA += 1
                    ax.scatter(x=i[0], y=i[1], color="#984ea3", marker=marker)
                else:
                    ax.scatter(x=i[0], y=i[1], color="#eeeeee", marker=marker)

        print("Superconducting atoms:", NS)
        print("Normal-metal atoms:", NN)
        print("Altermagnetic atoms:", NA)

        plt.ion()
        plt.show()


# %% Command-line arguments.
parser = ArgumentParser(description="Josephson junctions with altermagnetic interlayers")
parser.add_argument("-v", "--visualize", default=False, action="store_true")
parser.add_argument("-d", "--diagonal", default=False, action="store_true")
parser.add_argument("-l", "--length", type=int, required=True)
parser.add_argument("-m", "--field", type=float, required=True)

if interactive() is None:
    # Run as a script.
    args = parser.parse_args()
else:
    # Run interactively.
    # args = parser.parse_args(args=["-v", "-m 3", "-l 3"])
    args = parser.parse_args(args=["-d", "-v", "-m 3", "-l 3"])

print(args)

# %% Lattice preparation.
DIAG = bool(args.diagonal)
if DIAG:
    L_SC = 14
    L_Y = 14
else:
    L_SC = 20
    L_Y = 20

L_NM = 3
L_AM = args.length

L_X = 2 * L_SC + 2 * L_NM + L_AM
L_Z = 1

# Tight-binding parameters.
t = 1.0
Δ0 = 0.1 * t
μ = -0.5 * t
m = Δ0 * args.field

Tc = Δ0 / 1.764
T = 0.05 * Tc

# Lattice construction.
if DIAG:
    lattice = CubicLattice((L_X + L_Y, L_X + L_Y, 1))
else:
    lattice = CubicLattice((L_X, L_Y, 1))
visualize()

# %% Current calculation.
φs = np.linspace(0.0, 1.0, 51)
for δφ in tqdm(φs, desc="phase"):
    system = Hamiltonian(lattice)
    with system as (H, Δ, V):
        for i in lattice.sites():
            if IN(i):
                if SC1(i):
                    Δ[i, i] = Δ0 * jσ2 * np.exp((-1j / 2) * π * δφ)
                if SC2(i):
                    Δ[i, i] = Δ0 * jσ2 * np.exp((+1j / 2) * π * δφ)
                if AM(i):
                    H[i, i] = -μ * σ0 - m * σ3
                else:
                    H[i, i] = -μ * σ0
        for i, j in lattice.bonds(axis=0):
            if IN(i) and IN(j):
                H[i, j] = -t * σ0
        for i, j in lattice.bonds(axis=1):
            if IN(i) and IN(j):
                H[i, j] = -t * σ0

    J1x, J1y, J2x, J2y = current(system, T=T)
    print(f":: {DIAG},{L_SC},{L_NM},{L_AM},{m},{δφ},{J1x},{J1y},{J2x},{J2y}")

# %%
