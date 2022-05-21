#!/usr/bin/env python

"""Validation script to test the cutoff parameters.

This script runs a Local Chebyshev expansion of a simple tight-binding system,
and determines how the error scales as a function of the number of energies
(Chebyshev moments) and the cutoff radius (Local Krylov subspace).
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as spi
from scipy.sparse.linalg.matfuncs import _onenorm_matrix_power_nnm
from numpy.linalg import norm
from time import time

from bodge import *

# Physical parameters.
t = 1
Δ0 = 0.5
μ = 0.5
m3 = 0.25

# Numerical parameters.
energies = [128, 256]
radiuses = [1, 2, 4, 8, 16, 24, 32]
interval = np.linspace(-4, 4, 1024)

# Perform the validation.
if __name__ == "__main__":
    # Construct a 1D test system.
    lattice = CubicLattice((32, 32, 32))
    hamiltonian = Hamiltonian(lattice)
    with hamiltonian as (H, Δ):
        for i in lattice.sites():
            H[i, i] = -μ * σ0 - m3 * σ3
            Δ[i, i] = Δ0 * jσ2

        for i, j in lattice.bonds():
            H[i, j] = -t * σ0

    # Correct result.
    x = lattice[1, 1, 0]
    solver = Solver(chebyshev, hamiltonian, blocksize=16, energies=max(energies), radius=max(radiuses), resolve=True)
    ωs, ds = solver().density()
    exact = spi.pchip(ωs[::-1], ds[x, ::-1])(interval)
    def overlap(result):
        r1 = np.abs(result) / norm(result, 2)
        r2 = np.abs(exact) / norm(exact, 2)

        return np.dot(r1, r2)
    def error(result):
        return np.max(np.abs(result - exact))

    # Perform simulations.
    print("Energies,Radius,Time,Overlap,Error")
    for energy in energies:
        for radius in radiuses:
            # Instantiate solver.
            solver = Solver(chebyshev, hamiltonian, blocksize=128, energies=energy, radius=radius, resolve=True)

            # Calculate density of states.
            sec = time()
            ωs, ds = solver().density()
            sec = time() - sec

            # Interpolate density of states.
            result = spi.pchip(ωs[::-1], ds[x, ::-1])(interval)

            # Print out the estimated error.
            print(f"{energy},{radius},{sec},{overlap(result)},{error(result)}")