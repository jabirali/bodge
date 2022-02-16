import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("benchmark.csv")
df2 = pd.read_csv("benchmark2.csv")

# plt.figure()
# plt.loglog(df.N, df.Memory * (128/(8*1024*1024)), 'k.')
# plt.plot(df.N, df.N*1.5e-3, df.N, df.N*df.N*(128/(8*1024*1024)))
# plt.ylim([1, 1e5])
# plt.ylabel("Matrix memory usage [MB]")
# plt.xlabel("Number of lattice sites")

# plt.figure()
# plt.loglog(df.N, df.Skeleton, 'k.')
# plt.plot(df.N, df.N*4e-6)
# plt.ylabel("Matrix construction time [s]")
# plt.xlabel("Number of lattice sites")

# plt.figure()
# plt.loglog(df.N, df.Filling, 'k.')
# plt.plot(df.N, df.N*7e-5)
# plt.ylabel("Matrix filling time [s]")
# plt.xlabel("Number of lattice sites")

plt.figure()
plt.loglog(df.N, df.Multiplication, 'k.')
plt.plot(df.N, df.N*4e-6)
plt.plot(df2.N, df2.Matmul, 'r.')
plt.ylabel("Matrix multiplication time [s]")
plt.xlabel("Number of lattice sites")



plt.show()