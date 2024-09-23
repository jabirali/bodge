#!/usr/bin/env python

"""Benchmarking of Bodge vs. Kwant for constructing a Bogoliubov-deGennes Hamiltonian.

Note that this requires a `pip install kwant` to get the library benchmarked against.
"""

from time import sleep

import kwant
import numpy as np

from bodge import *

t = 1  # Hopping parameter
μ = -3 * t  # Chemical potential
M0 = 1.5 * t  # Magnetic exchange field
Δ0 = 0.1 * t  # Superconducting gap
χ = 0.5  # Superconducting phase winding


def timer(timers=[]):
    """Simple timer used for benchmarking."""
    from time import time

    # Save the current timestamp.
    timers.append(time())

    # Difference since last time stamp.
    try:
        return timers[-1] - timers[-2]
    except:
        return None


def bench_kwant(L, W, sparse=True):
    """Construct a Bogoliubov-deGennes Hamiltonian on a square lattice using Kwant."""
    # Define Pauli matrices for spin degree of freedom.
    σ_0 = np.array([[1, 0], [0, 1]])
    σ_1 = np.array([[0, 1], [1, 0]])
    σ_2 = np.array([[0, -1j], [1j, 0]])
    σ_3 = np.array([[1, 0], [0, -1]])

    # Define Pauli matrices for particle-hole degree of freedom.
    τ_0 = σ_0.copy()
    τ_1 = σ_1.copy()
    τ_2 = σ_2.copy()
    τ_3 = σ_3.copy()

    # Sleep before benchmarking.
    sleep(0.01)

    # Start a timer.
    timer()

    # Construct the lattice and Hamiltonian.
    lattice = kwant.lattice.square(norbs=4)
    system = kwant.Builder()

    # On-site terms in the Hamiltonian.
    for x in range(L):
        φ = χ * x / L
        τ_φ = 1j * (τ_2 * np.cos(φ) + τ_1 * np.sin(φ))

        for y in range(W):
            if x < L // 2:
                # Superconducting region.
                system[lattice(x, y)] = -μ * np.kron(τ_3, σ_0) - Δ0 * np.kron(τ_φ, 1j * σ_2)
            else:
                # Ferromagnetic region.
                system[lattice(x, y)] = -μ * np.kron(τ_3, σ_0) - M0 * np.kron(τ_3, σ_3)

    # Hopping terms in the Hamiltonian.
    for x in range(L - 1):
        for y in range(W):
            system[lattice(x, y), lattice(x + 1, y)] = -t * np.kron(τ_3, σ_0)
    for x in range(L):
        for y in range(W - 1):
            system[lattice(x, y), lattice(x, y + 1)] = -2 * t * np.kron(τ_3, σ_0)

    # Compile the Hamiltonian object to a matrix.
    system = system.finalized()
    H = system.hamiltonian_submatrix(sparse=sparse)

    # Benchmark results.
    T = timer()

    return H, T


def bench_bodge(L, W, sparse=True):
    """Construct a Bogoliubov-deGennes Hamiltonian on a square lattice using Bodge."""
    # Sleep before benchmarking.
    sleep(0.01)

    # Start a timer.
    timer()

    # Construct the lattice and Hamiltonian.
    lattice = CubicLattice((L, W, 1))
    system = Hamiltonian(lattice)

    # Fill out the Hamiltonian.
    with system as (H, Δ):
        # On-site terms.
        for i in lattice.sites():
            if i[0] < L // 2:
                # Superconducting region.
                H[i, i] = -μ * σ0
                Δ[i, i] = -Δ0 * np.exp(1j * χ * i[0] / L) * jσ2
            else:
                # Ferromagnetic region.
                H[i, i] = -μ * σ0 - M0 * σ3

        # Hopping terms.
        for i, j in lattice.bonds(axis=0):
            H[i, j] = -t * σ0
        for i, j in lattice.bonds(axis=1):
            H[i, j] = -2 * t * σ0

    # Compile the Hamiltonian object to a matrix.
    if sparse:
        H = system.matrix(format="csr")
    else:
        H = system.matrix(format="dense")

    # Benchmark results.
    T = timer()

    return H, T


if __name__ == "__main__":
    for L in [2**n for n in range(1, 12)]:
        for W in [L, 2 * L]:
            # Number of iterations. We repeat brief calculations to get better statistics.
            if L * W < 1e4:
                N = 30
            else:
                N = 1

            # Benchmark the matrix construction.
            T1s = []
            T2s = []
            for i in range(N):
                H1, T1 = bench_bodge(L, W)
                H2, T2 = bench_kwant(L, W)

                T1s.append(T1)
                T2s.append(T2)

            T1 = np.min(T1s)
            T2 = np.min(T2s)

            # Report the benchmarks.
            print(f"{L*W}\tBodge\t{T1}")
            print(f"{L*W}\tKwant\t{T2}")

            # Ensure the results are equal.
            print(f"{L*W}\tDiff\t{np.max(np.abs(H1 - H2))}")
