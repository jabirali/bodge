#!/usr/bin/env python

from pybdg import *

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

	solver = Solver(system)
	G = solver.run(8)

	print(G)