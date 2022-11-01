#!/usr/bin/env python


import csv
from time import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from bodge import *
from bodge.utils import ldos

Lx = 100
Ly = 100

t = 1
μ = 0.5

Δ_0 = 0.1 * t

lattice = CubicLattice((Lx, Ly, 1))
system = Hamiltonian(lattice)

# d = dvector("(e_x + je_y) * p_x")
# d = dvector("e_z * p_x")
# d = dvector("e_z * p_y")
# d = dvector("e_z * (p_x + jp_y)")
d = dvector("e_x * p_y - e_y * p_x")

with system as (H, Δ, V):
    for i in lattice.sites():
        H[i, i] = -μ * σ0
        # Δ[i, i] = -Δ_0 * jσ2

    for i, j in lattice.bonds():
        H[i, j] = -t * σ0
        Δ[i, j] = -Δ_0 * d(i, j)

sites = [
    (0, 0, 0),
    (Lx // 2, Ly // 2, 0),
    (0, Ly // 2, 0),
    (Lx // 2, 0, 0),
]

energies = np.linspace(0, 2 * Δ_0, 51)

t = time()
df = ldos(system, sites, energies)
print("\n", time() - t, "s")

# Plot the results.
grouped = df.groupby(["x", "y", "z"])
fig = plt.figure(figsize=(6, 6))
for n, (key, group) in enumerate(grouped):
    ax = fig.add_subplot(2, 2, n + 1)
    group.plot(ax=ax, x="ε", y="dos", color="k", legend=False)
    group.plot.area(ax=ax, x="ε", y="dos", color="k", alpha=0.3, legend=False)
    ax.title.set_text(key)
    ax.set_xlabel(r"Quasiparticle energy $ε/t$")
    ax.set_ylabel(r"Density of states $D(ε)$")
    ax.set_ylim([0, 3])
    ax.set_xlim([-energies[-1], +energies[-1]])

# plt.legend(title="Coordinate")
plt.tight_layout()
plt.show()
