#!/usr/bin/env python

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm, trange

from bodge import *
from bodge.utils import ldos, pwave

Lx = 64
Ly = 64
Lz = 1

lattice = CubicLattice((Lx, Ly, 1))

J0 = 3.0
Δ0 = 0.2
μ = -3.0

RKKY = {}

def calc_rkky(ds):
    for δ in trange(1, 25):
        for d in ds:
            D = pwave(d)

            i10 = 20
            i11 = Ly // 2
            i20 = i10 + δ
            i21 = i11

            # Construct the ferromagnetic Hamiltonian.
            system = Hamiltonian(lattice)
            with system as (H, Δ, V):
                for i in lattice.sites():
                    if i[0] == i10 and i[1] == i11:
                        H[i, i] = -μ * σ0 + (J0/2) * σ3
                    elif i[0] == i20 and i[1] == i21:
                        H[i, i] = -μ * σ0 + (J0/2) * σ3
                    else:
                        H[i, i] = -μ * σ0

                for i, j in lattice.bonds():
                    Δ[i, j] = -Δ0 * D(i, j)
                    H[i, j] = -1.0 * σ0

            # Calculate the corresponding free energy.
            E_fm = free_energy(system)

            # Construct the antiferromagnetic Hamiltonian.
            with system as (H, Δ, V):
                for i in lattice.sites():
                    if i[0] == i20 and i[1] == i21:
                        H[i, i] = -μ * σ0 - (J0/2) * σ3

            # Calculate the corresponding free energy.
            E_afm = free_energy(system)

            # Calculate the RKKY coupling.
            J_rkky = E_fm - E_afm

            # Print the results in CSV format.
            print(f"'{d}', {δ}, {J_rkky}")

            # Update the global dict.
            RKKY[(d, δ)] = np.array([J_rkky])

    # print(RKKY)

ds = [
    "e_z * p_x",
    "e_z * p_y",
    "e_z * (p_x + jp_y)",
    "(e_x + je_y) * (p_x + jp_y) / 2",
    "(e_x * p_x + je_y * p_y) / 2",
]

calc_rkky(ds)

# print(pd.DataFrame(RKKY))