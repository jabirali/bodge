#!/usr/bin/env python


import csv
from time import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from bodge import *
from bodge.utils import ldos

Lx = 100
Ly = 110

t = 1
μ = -1

Δ_0 = 0.1 * t

lattice = CubicLattice((Lx, Ly, 1))
system = Hamiltonian(lattice)

# d = dvector("(e_x + je_y) * p_x")
# d = dvector("e_z * p_x")
d = dvector("e_z * (p_x + jp_y)")

with system as (H, Δ, V):
    for i in lattice.sites():
        H[i, i] = -μ * σ0
        # Δ[i, i] = -Δ_0 * jσ2

    for i, j in lattice.bonds():
        H[i, j] = -t * σ0
        Δ[i, j] = -Δ_0 * d(i, j) / 2

sites = [
    (0, Ly // 2, 0),
    (Lx // 2, Ly // 2, 0),
    (Lx // 2, 0, 0),
]

# sites = [i for i in lattice.sites() if i[0] == 0]

energies = np.linspace(0, +2 * Δ_0, 51)

t = time()
df = ldos(system, sites, energies, 0.03 * Δ_0)
print("\n", time() - t, "s")

# Plot the results.
fig, ax = plt.subplots(figsize=(6, 6))
grouped = df.groupby(["x", "y", "z"])
for key, group in grouped:
    group.plot(ax=ax, x="ε", y="dos", label=key)

plt.legend(title="Lattice coordinate")
plt.ylim([0, None])
plt.show()
