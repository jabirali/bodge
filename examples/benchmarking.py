#!/usr/bin/env python

"""Benchmarking of Bodge vs. Kwant when constructing a Bogoliubov-deGennes Hamiltonian."""

import kwant
import numpy as np
from bodge import *

t = 1  # Hopping parameter
μ = -3 * t  # Chemical potential
M0 = 1.5 * t  # Magnetic exchange field
Δ0 = 0.1 * t  # Superconducting gap
χ = 0.1  # Superconducting phase winding


def timer(timers=[]):
    """Simple timer used for benchmarking."""
    from time import time

    # Save the current timestamp.
    timers.append(time())

    # Return the difference since last time stamp.
    try:
        return timers[-1] - timers[-2]
    except:
        return None


def bench_kwant(L, W, sparse=True):
    """Construct a Bogoliubov-deGennes Hamiltonian on a square lattice using Kwant."""
    # Import the required libraries.

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

    # Start a timer.
    timer()

    # Construct a square lattice and the Hamiltonian.
    lattice = kwant.lattice.square(norbs=4)
    system = kwant.Builder()

    # Fill out the terms in the Hamiltonian.
    for x in range(L):
        # Calculate the complex phase φ = πx/L of the order parameter at this point.
        φ = np.pi * x / L

        # The particle-hole structure of the pairing term depends on this phase.
        τ_φ = 1j * (τ_2 * np.cos(φ) + τ_1 * np.sin(φ))

        # Fill out the on-site contributions of the Hamiltonian.
        for y in range(W):
            if x < L // 2:
                # Superconducting region.
                system[lattice(x, y)] = -μ * np.kron(τ_3, σ_0) - Δ0 * np.kron(τ_φ, 1j * σ_2)
            else:
                # Ferromagnetic region.
                system[lattice(x, y)] = -μ * np.kron(τ_3, σ_0) - M0 * np.kron(τ_3, σ_3)

    # Hopping terms.
    for x in range(L - 1):
        for y in range(W):
            system[lattice(x, y), lattice(x + 1, y)] = -t * np.kron(τ_3, σ_0)
    for x in range(L):
        for y in range(W - 1):
            system[lattice(x, y), lattice(x, y + 1)] = -2 * t * np.kron(τ_3, σ_0)

    # Compile the Hamiltonian object to a matrix.
    system = system.finalized()
    H = system.hamiltonian_submatrix(sparse=sparse)

    # Report the benchmark results.
    print(f"Kwant time for {L}x{W} lattice: {timer()} seconds")

    return H


def bench_bodge(L, W, sparse=True):
    """Construct a Bogoliubov-deGennes Hamiltonian on a square lattice using Bodge."""
    # Start a timer.
    timer()

    # Construct the square lattice and the Hamiltonian.
    lattice = CubicLattice((L, W, 1))
    system = Hamiltonian(lattice)

    # Fill out the terms in the Hamiltonian.
    with system as (H, Δ):
        # On-site terms.
        for i in lattice.sites():
            if i[0] < L // 2:
                # Superconducting region.
                H[i, i] = -μ * σ0
                Δ[i, i] = -Δ0 * np.exp(1j * π * i[0] / L) * jσ2
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
        H = system(format="csr")
    else:
        H = system(format="dense")

    # Report the benchmark results.
    print(f"Bodge time for {L}x{W} lattice: {timer()} seconds")

    return H


if __name__ == "__main__":
    for N in [10, 30, 100, 300]:
        # Benchmark the matrix construction.
        H1 = bench_kwant(N, N)
        H2 = bench_bodge(N, N)

        # Ensure that the results are equal.
        print(f"Difference: {np.max(np.abs(H1 - H2))}")
