import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df = pd.read_csv("benchmark.dat", sep="\t")
print(df)
sns.lineplot(df, x="Atoms", y="Seconds", hue="Package", marker="o")

plt.xlim([1, 1e7])
plt.xscale("log")
plt.xlabel("Number of atoms")

plt.ylim([1e-3, 1e3])
plt.yscale("log")
plt.ylabel("Time to construct a sparse Hamiltonian (s)")

plt.show()
