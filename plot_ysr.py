#!/usr/bin/env python

# %% Imports
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['text.latex.preamble'] = r"\usepackage{newtxtext}\usepackage{newtxmath}"

# %% Data loading
dfs = pd.read_csv('ysr.csv')
display(dfs)

ds = {
    "e_z * p_x": r"$p_x$ wave",
    "e_z * p_y": r"$p_y$ wave",
    "e_z * (p_x + jp_y)": r"chiral $p$-wave",
    "(e_x + je_y) * (p_x + jp_y) / 2": r"non-unitary $p$-wave",
}
dfs['d'] = [ds[d] for d in dfs['d']]
display(dfs)

# %% Plotting
for n, d in enumerate(dfs.d.unique()):
    print(d)
    df = dfs[dfs.d == d]

    # Aggregation lists.
    x0s = []
    y0s = []

    x1s = []
    y1s = []

    # Extract δ>0.
    for δ in range(1, 6):
        df_ = df[(df['imp']) & (df['δ'] == δ)]
        x1s.append(df_['ε'])
        y1s.append(df_['dos'])

        df_ = df[(~df['imp']) & (df['δ'] == δ)]
        x0s.append(df_['ε'])
        y0s.append(df_['dos'])

    # Plot the extracted results.
    a = 0.75
    plt.figure(figsize=(3.375 / 1.25, 2 * 3.375))
    plt.gca().yaxis.set_ticks_position('none')
    for i, (x, y, x_, y_) in enumerate(zip(x1s, y1s, x0s, y0s)):
        plt.plot(x_, y_ - a*i, 'k')
        plt.plot(x, y - a*i, 'r')
        if i < 4:
            plt.plot(x, 0*x - a*i, 'k', lw=0.75)

    plt.xlim([-0.2, 0.2])
    plt.ylim([-4*a, a])
    if n == 0:
        plt.yticks([0.5 -a * i for i in range(5)], [r'$\delta = 1a$', r'$\delta = 2a$', r'$\delta = 3a$', r'$\delta = 4a$', r'$\delta = 5a$'], rotation=90)
    else:
        plt.yticks([0.2 -a * i for i in range(5)], ['' for _ in range(5)])
    plt.xticks([-0.1, 0, 0.1])
    plt.xlabel(r'Energy $\omega/t$', labelpad=10)
    plt.title(d)
    # plt.show()

    plt.savefig(f'ysr_{n}.pdf')

# %%
