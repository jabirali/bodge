#!/usr/bin/env python

"""Self-consistent calculation for s-wave superconductors."""

import csv

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm, trange

from bodge import *

# ------------------------------------------------------------
# Specify the physical system under investigation.
# ------------------------------------------------------------

# Physical parameters.
Lx = 100
Ly = 100

t = 1
μ = 0.1
U = 1.5

T = 1e-6 * t

# Numerical parameters.
R = None

# Non-superconducting Hamiltonian.
lattice = CubicLattice((Lx, Ly, 1))
system = Hamiltonian(lattice)
fermi = FermiMatrix(system, 1000)

with system as (H, Δ, V):
    for i in lattice.sites():
        H[i, i] = -μ * σ0
        V[i, i] = -U

    for i, j in lattice.bonds():
        H[i, j] = -t * σ0

# ------------------------------------------------------------
# Determine initial guess via geometric binary search.
# ------------------------------------------------------------
Δ_init = 0.06
# Δ_min = 1e-6
# Δ_max = 1
# for n in trange(12, desc="boot", unit="cyc", leave=False):
#     # Hamiltonian update.
#     Δ_init = np.sqrt(Δ_min * Δ_max)
#     with system as (H, Δ, V):
#         for i in lattice.sites():
#             if V[i, i] != 0:
#                 Δ[i, i] = Δ_init * jσ2

#     # Convergence control.
#     F = fermi(T)
#     Δ2 = np.abs(F.order_swave())
#     Δ1 = np.where(Δ2 > 0, Δ_init, 0)

#     if np.mean(Δ2) > np.mean(Δ1):
#         Δ_min = Δ_init
#     else:
#         Δ_max = Δ_init

# ------------------------------------------------------------
# Convergence via accelerated self-consistency iteration.
# ------------------------------------------------------------

with open("phase.csv", "w") as f:
    writer = csv.writer(f)

    for T in tqdm(np.linspace(1e-6, 2e-1, 100), desc="sweep", unit="tmp"):
        Δs = []
        for n in trange(300, desc="conv", unit="cyc", leave=False):
            # Order parameter update.
            Δs.append(fermi(T).order_swave())

            # Convergence control.
            if len(Δs) > 1:
                gap = np.real(np.mean(Δs[-1]))
                diff = np.max(np.abs(Δs[-1] - Δs[-2]))

                if diff < Δ_init * 1e-4:
                    break

            # Convergence acceleration.
            if len(Δs) > 4 and gap > Δ_init * 1e-2:
                Δs = [Δs[-3] - (Δs[-2] - Δs[-3]) ** 2 / (Δs[-1] - 2 * Δs[-2] + Δs[-3])]

            # Hamiltonian update.
            with system as (H, Δ, V):
                for i in lattice.sites():
                    Δ[i, i] = Δs[-1][i] * jσ2

            # Debug information.
            if len(Δs) > 1:
                print()
                print(f"Gap: {gap}")
                print(f"Diff: {diff}")

                plt.figure()
                plt.imshow(np.abs(Δs[-1]), vmin=0)
                plt.colorbar()
                plt.show()

        writer.writerow([T, gap, diff])
        f.flush()
