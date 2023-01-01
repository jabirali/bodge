#!/usr/bin/env python

# %% Common imports.
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.optimize import curve_fit

from bodge import *

# %% Load raw dataset.
data_raw = pd.read_csv('altermagnet.csv')

# %% Geometric transformations.
dfs = []
for ((m, diag), df) in data_raw.groupby(by=["m", "diag"]):
    if diag:
        # Diagonal junctions.
        df["L"] = df.L_AM * np.sqrt(2)
        df["J"] = ((df.J1x + df.J2x) / 2 - (df.J1y + df.J2y) / 2) / np.sqrt(2)
    else:
        # Non-diagonal junctions.
        df["L"] = df.L_AM
        df["J"] = (df.J1x + df.J2x) / 2

    dfs.append(pd.DataFrame(df))

data_rot = pd.concat(dfs)
display(data_rot)

# %% Fit current-phase relations to a sine series.
def sins(φ, *args):
    """Sine series with coefficients in *args."""
    J = 0.0
    for n, A_n in enumerate(args, 1):
        J += A_n * np.sin(n * π * φ)

    return J

def fits(df):
    """Fit dataframe to a sine series."""
    # Current-phase relation.
    φs = np.array(df.φ)
    js = np.array(df.J)

    # Perform curve fitting.
    ps = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ps, _ = curve_fit(sins, φs, js, ps)

    # Plot for comparison.
    # plt.plot(φs, js, 'r.', φs, [sins(φ, ps[0]) for φ in φs], 'b-')
    # plt.show()

    # Focus on 1st harmonic.
    return ps[0]

# Curve fit and construct new dataframe.
dfs = []
for ((L, m, diag), df) in data_rot.groupby(by=["L", "m", "diag"]):
    dfs.append([m, diag, L, fits(df)])
data_fit = pd.DataFrame(dfs, columns=["m", "d", "L", "A"])

# Normalize each series by A(L=0).
dfs = []
for (_, df) in data_fit.groupby(by=["m", "d"]):
    df.A = np.array(df.A) / np.array(df[df.L == 0].A)
    dfs.append(df)

data = pd.concat(dfs)

display(data)

# %% Visualize the data.
for (m, df) in data.groupby("m"):
    fig, ax = plt.subplots()
    sns.lineplot(data=df, x="L", y="A", hue="d", ax=ax)
    # plt.xlim([0, 40])
    # plt.ylim([-0.1,0.1])
    plt.show()

