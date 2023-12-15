#!/usr/bin/env python

"""Benchmarking of sparse matrix construction in Bodge vs. Kwant."""

from time import time

import kwant
import numpy as np

# # For plotting
# from matplotlib import pyplot

t = 1

L = 200
W = 200

μ = -3 * t
Δ0 = 0.1 * t
M0 = 1.5 * t

σ_0 = np.array([[1, 0], [0, 1]])
σ_1 = np.array([[0, 1], [1, 0]])
σ_2 = np.array([[0, -1j], [1j, 0]])
σ_3 = np.array([[1, 0], [0, -1]])

τ_0 = σ_0.copy()
τ_1 = σ_1.copy()
τ_2 = σ_2.copy()
τ_3 = σ_3.copy()

t = time()

# Construct an LxW square lattice.
lattice = kwant.lattice.square(norbs=4)
system = kwant.Builder()

# On-site potentials.
for x in range(L):
    for y in range(W):
        if x < L // 2:
            # Superconducting region.
            system[lattice(x, y)] = -μ * np.kron(τ_3, σ_0) - Δ0 * np.kron(1j * τ_2, 1j * σ_2)
        else:
            # Ferromagnetic region.
            system[lattice(x, y)] = -μ * np.kron(τ_3, σ_0) - M0 * np.kron(τ_3, σ_3)

# Hopping terms.
system[lattice.neighbors()] = -t * np.kron(τ_3, σ_0)

# Finalize the Hamiltonian.
system = system.finalized()

t = time() - t
print(f"Kwant object construction: {t}")

# Generate a sparse matrix.
t = time()
H_kwant_sparse = system.hamiltonian_submatrix(sparse=True)
t = time() - t
print(f"Kwant sparse construction: {t}")

# Generate a dense matrix.
t = time()
# H_kwant_dense = system.hamiltonian_submatrix(sparse=False)
t = time() - t
print(f"Kwant dense construction: {t}")

# Create a comparable Bodge setup.
from bodge import *

t = time()

lattice = CubicLattice((L, W, 1))
system = Hamiltonian(lattice)

with system as (H, Δ):
    for i in lattice.sites():
        if i[0] < L // 2:
            # Superconducting region.
            H[i, i] = -μ * σ0
            Δ[i, i] = -Δ0 * jσ2
        else:
            # Ferromagnetic region.
            H[i, i] = -μ * σ0 - M0 * σ3
    for i, j in lattice.bonds(axis=0):
        H[i, j] = -t * σ0
        # Δ[i, j] = Δ0 * jσ2
    for i, j in lattice.bonds(axis=1):
        H[i, j] = -t * σ0

t = time() - t
print(f"Bodge object creation: {t}")

t = time()
H_bodge_sparse = system(format="csr")
t = time() - t
print(f"Bodge sparse construction: {t}")

t = time()
# H_bodge_dense = system(format="dense")
t = time() - t
print(f"Bodge dense construction: {t}")

print(np.max(np.abs(H_bodge_sparse - H_kwant_sparse)))
print(np.max(np.abs(H_bodge_dense - H_kwant_dense)))
