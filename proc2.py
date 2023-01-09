#!/usr/bin/env python

# %% Common imports.
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import Rectangle

from scipy.optimize import curve_fit

from bodge import *

# %% Load raw dataset.
data_raw = pd.read_csv('altermagnetH.csv')

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
    plt.plot(φs, js, 'r.', φs, [sins(φ, ps[0]) for φ in φs], 'b-')
    plt.show()

    # Focus on 1st harmonic.
    return ps[0]

# Curve fit and construct new dataframe.
dfs = []
for ((m, diag, L), df) in data_rot.groupby(by=["m", "diag", "L"]):
    dfs.append([m, diag, L, fits(df)])
    print(dfs[-1])
data_fit = pd.DataFrame(dfs, columns=["m", "d", "L", "A"])

# Normalize each series by A(L=0).
dfs = []
for (_, df) in data_fit.groupby(by=["m", "d"]):
    print("A", _, float(df[df.L == 0].A))
    df.A = np.array(df.A) / np.array(df[df.L == 0].A)
    dfs.append(df)

data = pd.concat(dfs)

display(data)

# %% Visualize the data.
revtex()
l1, l2 = plt.plot([1,2,3], [4,5,6], [1,2,3], [4,5,6])

fig, axs = plt.subplots(1, 4, figsize=(2 * 3.375, (2.5 / 4) * 3.375), sharey=True)
for (m, df), ax in zip(data.groupby("m"), axs):
    ylim = [-0.1, 1.0]
    if m == 0.05:
        xlim = [0, 40]
        xcut = [12, 40]
        ycut = [-0.03, 0.06]

        ax.set_title(r"\textbf{(a)} $m = 0.5\Delta$")
    elif m == 0.15:
        xlim = [0, 40]
        xcut = [5, 40]
        ycut = [-0.08, 0.12]

        ax.set_title(r"\textbf{(b)} $m = 1.5\Delta$")
    elif m == 0.50:
        xlim = [0, 30]
        xcut = [1.5, 30]

        ax.set_title(r"\textbf{(c)} $m = 0.5t$")
    else:
        xlim = [0, 15]
        xcut = [1.5, 15]
        ycut = [-0.02, 0.02]

        ax.set_title(r"\textbf{(d)} $m = 0.9t$")
        # ax.set_xticks([0, 4, 8, 12, 16])
    ycut = [-0.08, 0.12]
    xcut = xlim

    ax.add_patch(Rectangle((xcut[0], ycut[0]), xcut[1] - xcut[0], ycut[1] - ycut[0], edgecolor='#ffd70040', facecolor='#ffd70040'))
    ax.axhline([0], color="#777777")
    ax.tick_params(axis="y", left="on", right="on")

    sns.lineplot(data=df, x="L", y="A", hue="d", ax=ax, legend=False)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_ylabel(r"First harmonic $I_1(L)/I_1(0)$")
    ax.set_xlabel(r"Altermagnet length $L/a$")
    ax.set_aspect((xlim[1] - xlim[0]) / (ylim[1] - ylim[0]))

    ins = inset_axes(ax, "60%", "60%" ,loc="upper right", borderpad=0.5)
    ins.axhline([0], color="#777777")
    ins.add_patch(Rectangle((xcut[0], ycut[0]), xcut[1] - xcut[0], ycut[1] - ycut[0], facecolor='#ffd70040'))

    sns.lineplot(data=df, x="L", y="A", hue="d", ax=ins, legend=False)

    ins.set_ylim(ycut)
    ins.set_xlim(xcut)
    ins.set_ylabel("")
    ins.set_xlabel("")
    ins.set_yticks([])
    ins.set_xticks([])

plt.figlegend([l1, l2], ["Straight junction", "Diagonal junction"], loc = 'upper center', ncol=2, labelspacing=2., bbox_to_anchor=(0.510,1.00), columnspacing=13.5)
plt.savefig("proc2.pdf", format="pdf")
plt.show()


# %% Extract critical current.
def crit(df):
    return np.max(np.abs(df.J))

# Calculate critical current.
dfs = []
for ((m, diag, L), df) in data_rot.groupby(by=["m", "diag", "L"]):
    dfs.append([m, diag, L, crit(df)])
    # print(dfs[-1])
data_crit = pd.DataFrame(dfs, columns=["m", "d", "L", "Ic"])

# Normalize each series by Ic(L=0).
dfs = []
for (_, df) in data_crit.groupby(by=["m", "d"]):
    print("Ic0", _, float(df[df.L == 0].Ic))
    df.Ic = np.array(df.Ic) / np.array(df[df.L == 0].Ic)
    dfs.append(df)

data_crit = pd.concat(dfs)

display(data_crit)

# %% Visualize critical current.
fig, ax = plt.subplots(figsize=(3.375, 0.66666 * 3.375))
df = data_crit[data_crit.m == 0.05]
sns.lineplot(data=df, x='L', y='Ic', hue='d', ax=ax)
ax.set_ylim([1e-3, 1])
ax.set_xlim([0, 40])
ax.set_yscale('log')
ax.set_xlabel(r'Altermagnet length $L$')
ax.set_ylabel(r'Critical current $I_c(L)/I_c(0)$')
ax.legend([l1, l2], ['Straight junction', 'Diagonal junction'])

plt.tight_layout()
plt.savefig("proc3.pdf", format="pdf")
# %%
