from math import log10

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('benchmark.csv')

x1 = np.arange(730, 8192, 1)
p1 = np.poly1d(np.polyfit(df['sites'][:-4], df['diag'][:-4], 3))
y1 = p1(x1)

p2 = np.poly1d(np.polyfit(np.log10(df['sites'][2:]), np.log10(df['cheb'][2:]), 1))
x2 = np.arange(730, 512*1024, 1)
y2 = 10**p2(np.log10(x2))

print(p1(32768))

print(x1)
print(y1)
plt.loglog(
	df['sites'], df['diag'], 'r.',
	df['sites'], df['cheb'], 'b.',
	x1, y1, 'r-',
	x2, y2, 'b-',
)
plt.legend(['Diagonalization (dense, parallel)', 'Local Chebyshev (sparse, parallel)'])
plt.minorticks_off()
plt.yticks([1, 8, 60, 8*60, 3600], ['1 sec', '8 sec', '1 min', '8 min', '1 hour'])
plt.xticks([100, 1000, 10000, 100000, 10**6])
plt.xlabel("Number of lattice sites")
plt.ylabel("Time per self-consistency iteration")
plt.title('Benchmark of BdG solvers for a real-space 3D S/F system')
plt.show()
