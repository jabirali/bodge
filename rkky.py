#!/usr/bin/env python

"""Calculate the generalized RKKY interactions on a superconductor.

The model system is two magentic impurities deposited on a superconductor. We
here wish to compare the responses of various p-wave superconductors with both
s-wave superconductors and normal metals. Moreover, we are here interested in
analyzing both the conventional contributions to the RKKY interaction, and
whether some DMI-like terms might arise e.g. in non-unitary situations.
"""

from bodge import *
from bodge.utils import pwave

# Define relevant physical parameters.
Lx, Ly, Lz = 64, 64, 1

t = 1.0
Δ0 = 0.2
J0 = 3.0
μ = -3.0

# Instantiate an appropriate square lattice.
lattice = CubicLattice((Lx, Ly, 1))

# Mapping of spin notation to σ matrices.
spins = {
    "+x": +σ1,
    "+y": +σ2,
    "+z": +σ3,
    "-x": -σ1,
    "-y": -σ2,
    "-z": -σ3,
}

# Mapping of d-vector notation to Δ matrices.
dvecs = {
    "N": None,
    "S": jσ2,
    "X": pwave("e_z * p_x"),
    "Y": pwave("e_z * p_y"),
    "CH": pwave("e_z * (p_x + jp_y)"),
    "NU1": pwave("(e_x + je_y) * (p_x + jp_y) / 2"),
    "NU2": pwave("(e_x * p_x + je_y * p_y) / 2"),
}

# Function for generating an RKKY Hamiltonian.
def hamiltonian(lattice, order, s1, s2, δ):
    system = Hamiltonian(lattice)
    with system as (H, Δ, _):
        # Prepare the usual hopping terms.
        for i, j in lattice.bonds():
            H[i, j] = -t * σ0

        # Place two impurities at given locations.
        S1 = spins[s1]
        x1 = 20 - 1
        y1 = Ly // 2

        S2 = spins[s2]
        x2 = x1 + δ + 1
        y2 = y1

        for i in lattice.sites():
            if i[0] == x1 and i[1] == y1:
                H[i, i] = -μ * σ0 - (J0 / 2) * S1
            elif i[0] == x2 and i[1] == y2:
                H[i, i] = -μ * σ0 - (J0 / 2) * S2
            else:
                H[i, i] = -μ * σ0

        # Prepare the superconducting contributions.
        D = dvecs[order]
        if D is not None:
            if not callable(D):
                # s-wave superconductivity.
                for i in lattice.sites():
                    Δ[i, i] = -Δ0 * D
            else:
                # p-wave superconductivity.
                for i, j in lattice.bonds():
                    Δ[i, j] = -Δ0 * D(i, j)

        return system

# Perform the actual simulations.
with open("rkky.csv", "w") as f:
    f.write("order,δ,s1,s2,energy\n")
    for δ in range(1, Lx - 20*2):
        for order in dvecs:
            for s1 in spins:
                for s2 in spins:
                    system = hamiltonian(lattice, order, s1, s2, δ)
                    energy = free_energy(system)
                    f.write(f"{order},{δ},{s1},{s2},{energy}\n")
                    f.flush()