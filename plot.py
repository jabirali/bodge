#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

fig, ax = plt.subplots(figsize=(4, 3), dpi=300)
df = pd.read_csv("giant.csv")

from scipy.optimize import curve_fit

plt.figure()
plt.plot(df.R[:47], df.t[:47], "b", df.R[47:], df.t[47:], "r")
plt.xlabel(r"Local Krylov cutoff radius $R$")
plt.ylabel(r"Time per self-consistency iteration [s]")
plt.tight_layout()
plt.show()


def f1(x, a, b):
    return a * np.exp(-x / b) + np.mean(df.Δ[-2:-1])


def f2(x, a, b):
    return a * np.exp(-x / b) + np.mean(df.Δ[-1:])


R1 = np.array(df.R[0::2])
Δ1 = np.array(df.Δ[0::2])
p1, _ = curve_fit(f1, R1, Δ1)

R2 = np.array(df.R[1::2])
Δ2 = np.array(df.Δ[1::2])
p2, _ = curve_fit(f2, R2, Δ2)


print(p1, p2)


plt.figure()
plt.plot(
    # df.R[0::2],
    # df.Δ[0::2],
    # "k-",
    # df.R[1::2],
    # df.Δ[1::2],
    df.R,
    f1(df.R, *p1),
    "k-",
    df.R,
    f2(df.R, *p2),
    "k-",
    # (df.Δ[-2] + df.Δ[-2]) / 2 + 0.1 / df.R,
    df.R,
    df.Δ,
    "r.",
)
# plt.
plt.ylim([0, 0.25])
plt.xlabel(r"Local Krylov cutoff radius $R$")
plt.ylabel(r"Order parameter at interface Δ[127, 15]")
plt.tight_layout()
plt.show()

# print()
# Δ0 = np.array(df[df.δ == 0].Δ)
# t0 = np.array(df[df.δ == 0].t)

# df = df[df.δ != 0]

# tol = np.array(df.δ)
# error = np.abs(np.array(df.Δ) - Δ0)
# speed = 1 / np.abs(np.array(df.t) / t0)
# # tol = np.array(df['tol']) / 4

# # df = df[df.Δ > 1e-6]

# # R = np.array(df['R'])
# # Δ = np.array(df['Δ'])
# # T2 = np.linspace(0, 0.035, 1000)

# # Tc = 0.0328
# # Δi = Δ[0] * np.tanh(1.74 * np.sqrt(Tc / T2 - 1))
# # Δi = np.where(Δi > 0, Δi, 0)

# plt.plot(tol, error, "k.")
# plt.ylabel(r"Error $[\Delta(\delta) - \Delta(0)]/t$")
# plt.xlabel(r"Local Krylov cutoff $\delta/t$")
# plt.ylim([1e-7, 1e-1])
# # plt.ylim([0, 0.3])
# # plt.xscale('log')
# # plt.yscale('log')
# plt.xscale("log")
# # plt.ylim([1e-6, 1e-1])
# # plt.xlim([1e-5, 1e-2])
# plt.yscale("log")
# # plt.ylim([0, 0.1])
# plt.tight_layout()
# plt.show()

# plt.figure()
# plt.plot(tol, speed, "k.")
# plt.ylabel(r"Relative speedup")
# plt.xlabel(r"Local Krylov cutoff $\delta/t$")
# # plt.ylim([1e-7, 1e-1])
# # plt.ylim([3e-1, 3e1])
# # plt.xscale('log')
# # plt.yscale('log')
# plt.xscale("log")
# # plt.ylim([1e-6, 1e-1])
# # plt.xlim([1e-5, 1e-2])
# # plt.yscale('log')
# # plt.ylim([0, 0.1])
# plt.tight_layout()
# plt.show()
# # print(df)

# plt.figure()
# plt.plot(1 / speed, error, "k.")
# # plt.xscale('log')
# plt.yscale("log")
# plt.tight_layout()
# plt.show()

# # Ns = []
# # Us = []
# # for N in df.N.unique():
# # 	Ns.append(N)
# # 	Us.append(np.min(df[df.N == N].U))

# # plt.plot(Ns, Us, 'k.', Ns, [(3.15-0.9*np.log10(n)) for n in Ns], 'k-')
# # plt.xscale('log')
# # plt.xticks([50, 100, 200, 400, 800], ['50', '100', '200', '400', '800'])
# # plt.ylim([0, 2])
# # plt.yticks([0, 0.5, 1.0, 1.5, 2.0], ['0.0', '0.5', '1.0', '1.5', '2.0'])
# # plt.xlabel(r'Chebyshev order $N$')
# # plt.ylabel(r'Minimum converging $|U|/t$')
# # plt.tight_layout()
# # plt.show()
# # # print(df)/

# # plt.yscale('log')
# # plt.ylim([1e-5, '1])
# # plt.xlabel(r"Coupling strength $-U/t$")
# # plt.ylabel(r"Order parameter $\Delta/t$")
# # plt.legend()
# # plt.tight_layout()
# # plt.show()
# # df.plot(x='U', y='Δ')

# # print(df)

# # # plt.plot(df.N, df.gap)

# # plt.yscale('log')
# # ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=5, frameon=False)
# # plt.tight_layout()
# # plt.show()
