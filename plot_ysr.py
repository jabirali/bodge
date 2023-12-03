#!/usr/bin/env python

# %% Imports
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

# %% Data loading
df = pd.read_csv('ysr.csv')
display(df)

ds = {
    "e_z * p_x": r"$p_x$ wave",
    "e_z * p_y": r"$p_y$ wave",
    "e_z * (p_x + jp_y)": r"chiral $p$-wave",
    "(e_x + je_y) * (p_x + jp_y) / 2": r"non-unitary $p$-wave",
}
df['d'] = [ds[d] for d in df['d']]
display(df)

# %% Plotting
# Aggregation lists.
x0s = []
y0s = []

x1s = []
y1s = []

# Extract δ>0.
for δ in range(1, 6):
    df_ = df[(df['imp']) & (df['δ'] == δ)]
    x1s.append(df_['ε'] * 10)
    y1s.append(df_['dos'])

    df_ = df[(~df['imp']) & (df['δ'] == δ)]
    x0s.append(df_['ε'] * 10)
    y0s.append(df_['dos'])

# Plot the extracted results.
a = 0.5
plt.figure(figsize=(3.375, 2*3.375))
for i, (x, y, x_, y_) in enumerate(zip(x1s, y1s, x0s, y0s)):
    plt.plot(x_, y_ - a*i, 'k')
    plt.plot(x, y - a*i, 'r')

plt.xlim([-2, 2])
plt.yticks([-1.8, -1.27, -0.83, -0.3, +0.18], ['δ = 5a', 'δ = 4a', 'δ = 3a', 'δ = 2a', 'δ = 1a'])
plt.xlabel(r'Quasiparticle energy $\omega/t$')
plt.title(df['d'][0])