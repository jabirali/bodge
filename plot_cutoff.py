#!/usr/bin/env python

"""Plot script for the accompanying validation script."""

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load the output data.
df = pd.read_csv(sys.argv[1])
energies = df["Energies"].unique()

# Plot the worst-case accuracy.
fig, ax = plt.subplots()
ax.set_xscale("log", base=2)
ax.set_yscale("log", base=10)
for energy in energies:
    dfe = df[df["Energies"] == energy]
    ax.plot(dfe["Radius"], dfe["Error"])

plt.xlim([2, 512])
plt.ylim([1e-8, 1])
plt.legend(
    [rf"$2^{{{int( np.log2(energy) )}}}$" for energy in energies],
    title="Energies",
    loc="lower left",
)
plt.xlabel("Cutoff Radius $R$")
plt.ylabel("DOS Max Absolute Error")
plt.show()

# Plot the average accuracy.
fig, ax = plt.subplots()
ax.set_xscale("log", base=2)
ax.set_yscale("log", base=10)
for energy in energies:
    dfe = df[df["Energies"] == energy]
    ax.plot(dfe["Radius"], (1 + 1e-16) - dfe["Overlap"])

plt.xlim([2, 512])
plt.ylim([1e-8, 1])
plt.legend(
    [rf"$2^{{{int( np.log2(energy) )}}}$" for energy in energies],
    title="Energies",
    loc="lower left",
)
plt.xlabel("Cutoff Radius $R$")
plt.ylabel("DOS Missing Overlap")
plt.show()
