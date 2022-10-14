#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import h5py
from tqdm import tqdm, trange

from bodge import *


# ------------------------------------------------------------
# Specify the physical system under investigation.
# ------------------------------------------------------------

# Physical parameters.
Lx = 100
Ly = 100

t = 1
μ = 0.1
U = 1.5

T = 1e-6 * t

# Numerical parameters.
R = None

# Non-superconducting Hamiltonian.
lattice = CubicLattice((Lx, Ly, 1))
system = Hamiltonian(lattice)
fermi = FermiMatrix(system, 1000)

with system as (H, Δ, V):
    for i in lattice.sites():
        H[i, i] = -μ * σ0
        V[i, i] = -U

    for i, j in lattice.bonds():
        H[i, j] = -t * σ0

with h5py.File('bodge.hdf5', 'w') as f:
    H = system.matrix.tocsc()
    f['hamiltonian/data'] = H.data
    f['hamiltonian/indices'] = H.indices
    f['hamiltonian/indptr'] = H.indptr
    f['hamiltonian/dim'] = H.shape[0]

# print(system.matrix)

with h5py.File('bodge.hdf5') as f2:
    print(f2['hamiltonian/data'][...])