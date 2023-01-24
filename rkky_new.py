#!/usr/bin/env python

"""RKKY interaction between magnetic impurities on a superconductor.

Usage:
    rkky_new.py (-h | --help)
    rkky_new.py
        [--length=<Lx>] [--width=<Ly>] [--offset=<d1>] [--separation=<d2>]
        [--axis1=<s1>] [--axis2=<s2>] [--mag=<J0>] [--gap=<Δ0>] [--pot=<μ>]

Options:
    -h --help           Show this screen
    --length=<Lx>       Lattice length [default: 64]
    --width=<Ly>        Lattice width [default: 64]
    --offset=<d1>       Distance from edge to left impurity [default: 20]
    --separation=<d2>   Distance between the two impurities [default: 20]
    --axis1=<s1>        Spin axis of the left impurity [default: +z]
    --axis2=<s2>        Spin axis of the right impurity [default: +z]
    --mag=<J0>          Coupling to impurity spin [default: 3.0]
    --gap=<Δ0>          Superconducting gap [default: 0.2]
    --pot=<μ>           Chemical potential [default: -3.0]
"""

from docopt import docopt

# from click import command, option
from icecream import ic

from bodge import *

if __name__ == "__main__":
    arguments = docopt(__doc__)
    print(arguments)


# # @command()
# # @option("--Lx", "Lx", default=81, type=int)
# # @option("--Ly", "Ly", default=81, type=int)
# # @option("--μ0", "μ0", default=-3.0, type=float)
# # @option("--Δ0", "Δ0", default=0.2, type=float)
# # @option("--J0", "J0", default=3.0, type=float)
# # @option("--s1", "s1", default="+z", type=str)
# # @option("--s2", "s2", default="+z", type=str)
# # @option("--d1", "d1", default=20, type=int)
# # @option("--d2", "d2", default=20, type=int)
# def main(Lx, Ly, μ0, Δ0, J0, s1, s2, d1, d2):
#     # Square lattice.
#     lattice = CubicLattice((Lx, Ly, 1))
#     ic(lattice.shape)

#     # Impurity sites.
#     x1 = d1
#     y1 = Ly // 2
#     z1 = 0

#     x2 = d1 + d2 + 1
#     y2 = y1
#     z2 = z1

#     if x2 > Lx - d1:
#         raise RuntimeError("External distance requirements violated.")

#     i1 = (x1, y1, z1)
#     i2 = (x2, y2, z2)

#     ic(i1)
#     ic(i2)

#     # Impurity spins.
#     spins = {
#         "+x": +σ1,
#         "+y": +σ2,
#         "+z": +σ3,
#         "-x": -σ1,
#         "-y": -σ2,
#         "-z": -σ3,
#     }

#     S1 = spins[s1]
#     S2 = spins[s2]

#     ic(S1)
#     ic(S2)

#     # Construct the Hamiltonian.
#     system = Hamiltonian(lattice)
#     with system as (H, Δ, _):
#         for i in lattice.sites():
#             Δ[i, i] = -Δ0 * jσ2
#             if i == i1:
#                 H[i, i] = -μ0 * σ0 - J0 * S1
#             elif i == i2:
#                 H[i, i] = -μ0 * σ0 - J0 * S2
#             else:
#                 H[i, i] = -μ0 * σ0

#         for i, j in lattice.bonds():
#             H[i, j] = -σ0

#     # Critical temperature.
#     Tc = Δ0 / 1.764

#     # Calculate the free energy.
#     F = free_energy(system, 0.01 * Tc)

#     # Save the results.
#     with open("rkky.csv", "a+") as f:
#         f.write(f"{s1}, {s2}, {d2}, {F}\n")


# # Run `main` as a script.
# if __name__ == "__main__":
#     main()
