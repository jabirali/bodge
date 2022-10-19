#!/usr/bin/env python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots(figsize=(4, 3), dpi=300)
df = pd.read_csv("chebpols.csv")

plt.figure()
plt.plot([0, *df.N/1000], [0, *df.Tc])
plt.xlim([0, 14])
plt.xticks([0, 2, 4, 6, 8, 10, 12, 14])
plt.xlabel(r"Chebyshev order $N/1000$")
plt.ylabel(r"Critical temperature $T_c/t$")
plt.tight_layout()
plt.show()

plt.figure()
plt.plot([0, *df.N/1000], [0, *df.Δ0])
plt.xlim([0, 4])
plt.xticks([0, 1, 2, 3, 4])
plt.xlabel(r"Chebyshev order $N/1000$")
plt.ylabel(r"Zero-temperature gap $Δ_0/t$")
plt.tight_layout()
plt.show()

plt.figure()
plt.plot([*df.N/1000], [*(1 + (0.01*2/1.764)/2**13 - df.Tc/np.array(df.Tc)[-1])])
plt.xlim([0, 14])
plt.ylim([1e-3, 1])
plt.yscale('log')
plt.xticks([0, 2, 4, 6, 8, 10, 12, 14])
plt.xlabel(r"Chebyshev order $N/1000$")
plt.ylabel(r"Critical temperature error $1 - T_c(N)/T_c(14,000)$")
plt.grid()
plt.tight_layout()
plt.show()

plt.figure()
plt.plot([*df.N/1000], (0.03/2**13 + np.abs(1-df.Δ0/0.010733642578124997)))
plt.xlim([0, 5])
plt.ylim([1e-4, 1])
plt.yscale('log')
plt.xticks([0, 1, 2, 3, 4, 5])
plt.xlabel(r"Chebyshev order $N/1000$")
plt.ylabel(r"Zero-temperature gap error $1 - Δ_0(N)/Δ_0(4,800)$")
plt.grid()
plt.tight_layout()
plt.show()
