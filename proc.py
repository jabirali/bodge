#!/usr/bin/env python

"""Visualization script for simulations of altermagnetic Josephson junctions."""

# %% Common imports.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import curve_fit

from bodge import *

# %% Load dataset.
df = pd.read_csv("test2.csv")
df = df[df.m == 0.05]

DIAG = False
df = df[df.diag == DIAG]

df


# NOTE: Find that we in general cannot fit to an exp(L/ξ) cos(L/ξ) relationship, so qualitatively different.
# The results in general display 0-π oscillations first *after* the current has decayed.

# NOTE: We also find that for m = 0.9t, the difference between the two orientations become strikingly higher.

# %%
# df1 = df[(df.m == 0.15) & (df.diag == False)]
# df1 = df[(df.m == 0.05) & (df.diag == True)]
# df1["J"] = (df1.J1x + df1.J2x) / 2
# df1['J'] = (df1.J1x - df1.J1y)/np.sqrt(2)
# df1["δ"] = np.abs(df1.J1x - df1.J2x) + np.abs(df1.J1y) + np.abs(df1.J2y)

# for L_AM in df1.L_AM.unique():
#     df1_ = df1[df1.L_AM == L_AM]
#     df1_

if DIAG:
    df["J"] = (df.J1x + df.J2x) / 2
else:
    df["J"] = (df.J1x + df.J2x - df.J1y - df.J2y) / (2 * np.sqrt(2))

# df1 = df1.pivot(index='φ', columns='L_AM', values='J').groupby('L_AM')
# df1 = df1.groupby('L_AM')


def sinish(φ, *args):
    J = 0.0
    for n, a_n in enumerate(args, 1):
        J += a_n * np.sin(n * π * φ)

    return J


Ls = []
Ps = []
Cs = []
for L_AM, dfL in df.groupby("L_AM"):
    # Extract the current-phase relation.
    φs = dfL.φ
    js = dfL.J

    # Perform the curve fitting action.
    ps = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ps, _ = curve_fit(sinish, φs, js, ps)
    fs = np.array([sinish(φ, *ps) for φ in φs])

    # Check the curve fitting error.
    print("error:", np.std(fs - js) / np.max(np.abs(js)))

    # Extract the relevant metrics.
    print("1st harmonic:", ps[0], L_AM)
    print("all harmonics:", [p/ps[0] for p in ps])
    Ls.append(L_AM)
    Ps.append(ps[0])
    Cs.append(np.max(np.abs(js)))

    # Visualize the current fit.
    plt.plot(φs, js, "k.", φs, fs, "r-")
    plt.show()

# def decay(x, A, b, c):
#     return A * np.exp(-b*x) * np.cos(b*x+c)
# qs, _ = curve_fit(decay, Ls[1:], [P/Cs[0] for P in Ps[1:]])

# if DIAG:
#     plt.plot(Ls, [1e2 * P / Cs[0] for P in Ps], Ls, [1e2 * decay(L, *qs) for L in Ls])
#     plt.plot(Ls, [1e2 * P / Cs[0] for P in Ps], Ls, [1e2 * decay(L, *qs) for L in Ls])
# else:
#     plt.plot(Ls, [1e3 * P / Cs[0] for P in Ps], Ls, [1e3 * decay(L, *qs) for L in Ls])
if DIAG:
    plt.plot([L * np.sqrt(2) for L in Ls], [P / Cs[0] for P in Ps])
else:
    plt.plot(Ls, [P / Cs[0] for P in Ps])
plt.xlim([0, None])
# plt.ylim([-2, 2])
plt.grid()
plt.show()

if DIAG:
    plt.plot([L * np.sqrt(2) for L in Ls], [C / Cs[0] for C in Cs])
else:
    plt.plot([L for L in Ls], [C / Cs[0] for C in Cs])
plt.xlim([0, None])
plt.yscale("log")
plt.show()
# plt.show()
# sns.lineplot(data=dfL, x='φ', y='J')
# sns.heatmap(data=df1, center=0)

# φs = [φ for φ in df.φ.unique()]
# Ls = [L_AM for L_AM in df.L_AM.unique()]

# df1 = df[(df.)]
# sns.lineplot(data=df, x="L_AM", y="")
# df

# # %%
# def Ic(df):
#     φs = [φ for φ in df.φ.unique() if φ <= 1]
#     Ls = [L_AM for L_AM in df.L_AM.unique()]
#     Js = []

#     for L in Ls:
#         J = 0
#         for φ in φs:
#             try:
#                 j = float(df[(df.φ == φ) & (df.L_AM == L)].J)
#                 if np.abs(j) > np.abs(J):
#                     J = j
#             except:
#                 continue

#         Js.append(J)

#     return Ls, Js


# df1 = df[ (df.m == 0.5 * 0.05) & (df.diag == False)]
# df2 = df[ (df.m == 0.5 * 0.05) & (df.diag == True)]
# df3 = df[ (df.m == 0.075) & (df.diag == False)]
# df4 = df[ (df.m == 0.075) & (df.diag == True)]
# print(df.m.unique())
# # df3 = df[ (df.m == 0.00) & (df.diag == False)]
# # df4 = df[ (df.m == 0.00) & (df.diag == True)]
# # df5 = df[ (df.m == 0.95) & (df.diag == False)]
# # df6 = df[ (df.m == 0.95) & (df.diag == True)]
#
# fig, ax = plt.subplots()
#
# Ls, Js = Ic(df1)
# ax.plot(Ls, Js)
#
# Ls, Js = Ic(df2)
# Ls = [L * np.sqrt(2) for L in Ls]
# ax.plot(Ls, Js)
#
# print(df1[df1.φ == 0.5])
# print(df2[df2.φ == 0.5])
# plt.show()
#
# fig, ax = plt.subplots()
#
# Ls, Js = Ic(df3)
# Ls = [L for L in Ls]
# ax.plot(Ls, Js)
#
# Ls, Js = Ic(df4)
# Ls = [L * np.sqrt(2) for L in Ls]
# ax.plot(Ls, Js)
#
# print(df3[df3.φ == 0.5])
# print(df4[df4.φ == 0.5])
#
# plt.show()

# %%
