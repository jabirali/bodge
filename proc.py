#!/usr/bin/env python

"""Visualization script for simulations of altermagnetic Josephson junctions."""

# %% Common imports.
from bodge import *
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# %% Load dataset.
df = pd.read_csv('test2.csv')
df

# %% 
df1 = df[(df.m == 0.15) & (df.diag == False)]
# df1 = df[(df.m == 0.05) & (df.diag == True)]
df1['J'] = (df1.J1x + df1.J2x)/2
# df1['J'] = (df1.J1x - df1.J1y)/np.sqrt(2)
df1['δ'] = np.abs(df1.J1x - df1.J2x) + np.abs(df1.J1y) + np.abs(df1.J2y)

# for L_AM in df1.L_AM.unique():
#     df1_ = df1[df1.L_AM == L_AM]
#     df1_

# df1 = df1.pivot(index='φ', columns='L_AM', values='J').groupby('L_AM')
# df1 = df1.groupby('L_AM')

def sinish(φ, *args):
    J = 0.0
    for n, a_n in enumerate(args, 1):
        J += a_n * np.sin(n*π*φ)
    
    return J

Ls = []
Ps = []
Cs = []
for L_AM, dfL in df1.groupby('L_AM'):
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
    print("first harmonic:", ps[0])
    Ls.append(L_AM)
    Ps.append(ps[0])
    Cs.append(np.max(np.abs(js)))

    # Visualize the current fit.
    plt.plot(φs, js, 'k.', φs, fs, 'r-')
    plt.show()

plt.plot(Ls, [P/Cs[0] for P in Ps])
plt.xlim([0, None])
# plt.ylim([-0.1, 0.1])
plt.grid()
plt.show()

plt.plot(Ls, Cs)
plt.yscale('log')
plt.show()
    # plt.show()
    # sns.lineplot(data=dfL, x='φ', y='J')
# sns.heatmap(data=df1, center=0)

# φs = [φ for φ in df.φ.unique()]
# Ls = [L_AM for L_AM in df.L_AM.unique()]

# df1 = df[(df.)]
# sns.lineplot(data=df, x="L_AM", y="")
# df 

# %% 
def Ic(df):
    φs = [φ for φ in df.φ.unique() if φ <= 1]
    Ls = [L_AM for L_AM in df.L_AM.unique()]
    Js = []

    for L in Ls:
        J = 0
        for φ in φs:
            try:
                j = float(df[(df.φ == φ) & (df.L_AM == L)].J)
                if np.abs(j) > np.abs(J):
                    J = j
            except:
                continue
            
        Js.append(J)

    return Ls, Js



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
