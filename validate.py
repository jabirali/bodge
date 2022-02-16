#!/usr/bin/env python

"""
This is a test script that constructs a simple tight-binding Hamiltonian for
a superconducting system and subsequently calculates the density of states.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from scipy.linalg import inv
from tqdm import tqdm

from pybdg import *


t = 1.0
μ = +3*t
Δ0 = t/2
m3 = t/5

lattice = Cubic((30, 10, 30))
system = System(lattice)
with system as (H, Δ):
	for i in lattice.sites():
		H[i, i] = -μ * σ0 - m3 * σ3
		Δ[i, i] = Δ0 * jσ2

	for i, j in lattice.neighbors():
		H[i, j] = -t * σ0