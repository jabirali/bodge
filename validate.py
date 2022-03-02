#!/usr/bin/env python

"""
This is a test script that constructs a simple tight-binding Hamiltonian for
a superconducting system and subsequently calculates the density of states.
"""
from multiprocessing import Pool
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from tqdm import tqdm

from pybdg import *




# X = system.diagonalize()
# Y = system.spectralize([0.0, 1.0, 2.0])

if __name__ == "__main__":
	t = 1.0
	μ = +3*t
	Δ0 = t/2
	m3 = t/5

	lattice = Cubic((64, 64, 8))
	system = System(lattice)

	with system as (H, Δ):
		for i in lattice.sites():
			H[i, i] = -μ * σ0 - m3 * σ3
			Δ[i, i] = Δ0 * jσ2

		for i, j in lattice.neighbors():
			H[i, j] = -t * σ0

	kernel, blocks = system.chebyshev(200)
	with Pool(4) as p:
		G = p.map(kernel, blocks)

	print(G)