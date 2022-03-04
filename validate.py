#!/usr/bin/env python

import sys
from pybdg import *

t = 1.0
μ = +3*t
Δ0 = t/2
m3 = t/5

lattice = Cubic((32, 32, 8))
system = System(lattice)

with system as (H, Δ):
	for i in lattice.sites():
		H[i, i] = -μ * σ0 - m3 * σ3
		Δ[i, i] = Δ0 * jσ2

	for i, j in lattice.neighbors():
		H[i, j] = -t * σ0

# print(system.diagonalize())
solver = Solver(system)
G = solver.run(7)
# G = solver.run(7)

# print(G)