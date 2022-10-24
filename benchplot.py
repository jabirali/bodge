#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import curve_fit

df = pd.read_csv("bench_all.csv", sep=",")


def f2(x, a):
    return a * x**4


def f2e(x, a, b):
    return a * x**4 + b


def f3(x, a):
    return a * x**6


def f3e(x, a):
    return a * x**5


def f4e(x, a):
    return a * 2 ** (x / 6.0)


df["t"] = df.groupby(["L", "M"])["t"].transform("min")
df.drop_duplicates(inplace=True)

fig, ax = plt.subplots()
for method in df.M.unique():
    L = np.array(df[df.M == method].L)
    t = np.array(df[df.M == method].t)
    L2 = np.linspace(1, 160, 100)

    match method:
        case "nagai":
            plt.plot(L, t, ".", color="#f4a582", label=r"Nagai (SciPy, $N = 1024$)")
        case "cheb":
            plt.plot(L, t, ".", color="#ca0020", label=r"Chebyshev (SciPy, $N = 1024$)")
            p, _ = curve_fit(f2, L[-18:], t[-18:])
            plt.plot(L2, f2(L2, *p), "-", color="#ca0020", label=None)
            plt.annotate(r"  $\mathcal{O}(L^4)$", (160, 2.7 * 60 * 60))
        case "diag":
            plt.plot(L, t, ".", color="#92c5de", label=r"Diagonalization (SciPy)")
            p, _ = curve_fit(f4e, L[24:], t[24:])
            print(p)
            plt.plot(L2, f4e(L2, *p), "-", color="#92c5de", label=None)
            plt.annotate(r"$\mathcal{O}(2^{L /6})$", (97, 26 * 3600))
        case "eigd":
            plt.plot(L, t, ".", color="#0571b0", label=r"Eigenvalues (SciPy)")
            p, _ = curve_fit(f3, L[10:], t[10:])
            plt.plot(L2, f3(L2, *p), "-", color="#0571b0", label=None)
            plt.annotate(r"  $\mathcal{O}(L^6)$", (160, 13 * 3600))
        case "eigd_intel":
            plt.plot(L, t, ".", color="#5e3c99", label=r"Eigenvalues (Intel MKL)")
            p, _ = curve_fit(f3e, L[10:], t[10:])
            plt.plot(L2, f3e(L2, *p), "-", color="#5e3c99", label=None)
            plt.annotate(r"  $\mathcal{O}(L^6)$", (160, 1.3 * 60 * 60))
            # p, _ = curve_fit(f3e, L[10:], t[10:])
            # plt.plot(L2, f3e(L2, *p), '-', color='#5e3c99', label = None)

    # if method == 'cheb':
    # 	p, _ = curve_fit(f2, L, t)
    # 	plt.plot(L2, f2(L2, *p), colors[method] + '-', label = None)
    # if method == 'eigd':
    # 	p, _ = curve_fit(f3, L, t)
    # 	plt.plot(L2, f3(L2, *p), colors[method] + '-', label = None)
# df.groupby('M')['L', 't'].plot(x = 'L', y = 't')

# L = np.arange(1, 150, 100)
# plt.plot(L, 0.5*L**4)

# plt.xlabel(r"")
# plt.ylabel(r"")
# ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=5, frameon=False)
plt.legend()
plt.grid()
plt.title(r"$L×L$ lattice on a node with 52 CPU cores and 175 GB RAM")
plt.yscale("log")
plt.yticks(
    [1, 4, 15, 60, 4 * 60, 15 * 60, 60 * 60, 4 * 60 * 60, 15 * 60 * 60, 2 * 24 * 60 * 60],
    [
        "1 sec",
        "4 sec",
        "15 sec",
        "1 min",
        "4 min",
        "15 min",
        "1 hour",
        "4 hours",
        "15 hours",
        "48 hours",
    ],
)
plt.xticks([16 * n for n in range(200 // 16)])
plt.xlim([16, 160])
plt.ylim([1, 2 * 24 * 60 * 60])
plt.xlabel(r"Lattice scale $L/a$")
plt.ylabel("Time per self-consistency iteration")
# plt.xscale('log')
plt.tight_layout()
plt.show()
