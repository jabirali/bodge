#!/usr/bin/env python

"""Benchmark different solvers."""

import csv
from time import time

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm, trange

from bodge import *

t = 1
μ = 0.1
U = 1.0
δ = 0.06
T = 1e-2

def fe(x):
    return (1 - np.tanh((4 * t * x) / 2)) / 2

def entropy(x):
    return np.sum(np.log(fe(x)))

with open("bench.csv", "w") as f:
    writer = csv.writer(f)
    
    for L in trange(2, 201, 2):
        # Non-superconducting Hamiltonian.
        lattice = CubicLattice((L, L, 1))
        system = Hamiltonian(lattice)
        fermi = FermiMatrix(system, 1024)

        with system as (H, Δ, V):
            for i in lattice.sites():
                H[i, i] = -μ * σ0
                Δ[i, i] = -δ * jσ2 
                V[i, i] = -U

            for i, j in lattice.bonds():
                H[i, j] = -t * σ0


        t_cheb = time()
        Δ = np.mean(fermi(T).order_swave())
        t_cheb = time() - t_cheb
        writer.writerow([L, 'cheb', t_cheb])
        f.flush()

        t_eigs = time()
        E1 = system.eigenvalues(method='dense')
        t_eigs = time() - t_eigs
        writer.writerow([L, 'eigd', t_eigs])
        f.flush()

        # t_eigs = time()
        # E2 = system.eigenvalues(method='sparse')
        # t_eigs = time() - t_eigs
        # writer.writerow([L, 'eigs', t_eigs])
        # f.flush()

        # writer.writerow([L, 'serr', 1 - entropy(E2)/entropy(E1)])
        writer.writerow([])



