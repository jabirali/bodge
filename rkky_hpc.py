#!/usr/bin/env python

"""Calculate the generalized RKKY interactions on a superconductor.

The model system is two magentic impurities deposited on a superconductor. We
here wish to compare the responses of various p-wave superconductors with both
s-wave superconductors and normal metals. Moreover, we are here interested in
analyzing both the conventional contributions to the RKKY interaction, and
whether some DMI-like terms might arise e.g. in non-unitary situations.
"""

import numpy as np
import pandas as pd
from tqdm import tqdm

from argparse import ArgumentParser

from bodge import *
from bodge.utils import ldos, pwave

# Fetch and print command-line arguments.
parser = ArgumentParser()
parser.add_argument("--len", default=64, type=int, help="Side length for an LxL lattice (L/a)")
parser.add_argument("--gap", default=+0.20, type=float, help="Superconducting order parameter (Δ0/t)")
parser.add_argument("--imp", default=+3.00, type=float, help="Coupling to impurity spins (J0/t)")
parser.add_argument("--pot", default=-3.00, type=float, help="Chemical potential (μ/t)")
parser.add_argument("--sep", default=1, type=int, help="Distance between impurities (δ/a)")
args = vars(parser.parse_args())

print("Parameters:")
for key, val in args.items():
    print(f"--{key} {val}")

# Extract these parameters.
L = args['len']
δ = args["sep"]
µ = args["pot"]
J0 = args["imp"]
Δ0 = args["gap"]

# Instantiate a square lattice.
lattice = CubicLattice((L, L, 1))

# Instantiate a new non-superconducting Hamiltonian.
def normal():
    t = 1.0
    system = Hamiltonian(lattice)
    with system as (H, Δ, _):
        for i in lattice.sites():
            H[i, i] = -μ * σ0

        for i, j in lattice.bonds():
            H[i, j] = -t * σ0
        
    return system

# Define the mapping of notation to spins.
spins = {
    "+x": +σ1, "+y": +σ2, "+z": +σ3, 
    "-x": -σ1, "-y": -σ2, "-z": -σ3, 
}

# Create two magnetic impurities in the system,
# which have spins oriented along s1 and s2.
def impurities(system, s1, s2):
    # Use this to rewrite inputs in terms of Pauli matrices.
    S1 = spins[s1], S2 = spins[s2]

    # Determine the corresponding coordinates for impurities.
    x1 = 20
    y1 = L // 2

    x2 = x1 + 1 + δ
    y2 = y1

    # Fill the corresponding Hamiltonian elements.
    with system as (H, Δ, _):
        H[(x1, y1, 0), (x1, y1, 0)] = -μ * σ0 + (J0/2) * S1
        H[(x2, y2, 0), (x2, y2, 0)] = -μ * σ0 + (J0/2) * S2

# Create superconducting order parameters.
def swave(system):
    with system as (H, Δ, _):
        for i in lattice.sites():
            Δ[i, i] = -Δ0 * jσ2

def pwave(system, dvec):
    D = pwave(dvec)
    with system as (H, Δ, _):
        for i, j in lattice.bonds():
            Δ[i, j] = -Δ0 * D(i, j)

# Completely normal metal.
for s1 in spins:
    for s2 in spins:
        system = normal() 
        impurities(system, s1, s2)
        energy = free_energy(system)

        print(f"normal,{s1},{s2},{energy}")

# # Conventional superconductor.
# system = new() 
# swave(system)
# E = free_energy(system)
# print(f"s-wave,{s1},{s2},{E}")

# E = free_energy(system)