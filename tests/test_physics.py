"""Integration tests that validate some known physical phenomena using Bodge.

In contrast to `test_lattice.py` and `test_hamiltonian.py`, these are
not intended to test the mathematical properties of individual classes
or methods, but rather to combine features from across Bodge in order
to verify that important physical phenomena are modeled correctly.
"""

import numpy as np
from numpy.random import random as r

from bodge import *
from bodge.common import *


def test_superconducting_gap_existence():
    """When a normal metal becomes superconducting a "gap" opens.

    There's two ways to test this:

    1. The local density of states disappears within the gap;
    2. The lowest energy eigenvalue is pushed above the gap.

    Here, we verify this result using both approaches.
    """
    # Instantiate a normal metal.
    lattice = CubicLattice((16, 16, 1))
    system = Hamiltonian(lattice)

    with system as (H, Δ):
        for i in lattice.sites():
            H[i, i] = -1.5 * σ0
        for i, j in lattice.bonds():
            H[i, j] = -1.0 * σ0

    # Calculate the central LDOS.
    Δs = 0.5
    i = (8, 8, 0)
    ω = np.array([-1.2 * Δs, -0.8 * Δs, +0.8 * Δs, 1.2 * Δs])
    ρ1 = system.ldos(i, ω)

    # Calculate the lowest eigenvalue.
    ε, _ = system.diagonalize()
    ε1 = np.min(ε)

    # Add superconductivity.
    with system as (H, Δ):
        for i in lattice.sites():
            Δ[i, i] = Δs * jσ2

    # Calculate the central LDOS.
    ρ2 = system.ldos(i, ω)

    # Calculate the lowest eigenvalue.
    ε, _ = system.diagonalize()
    ε2 = np.min(ε)

    # The LDOS should decrease inside the gap of a superconductor.
    assert ρ2[1] < ρ1[1]
    assert ρ2[2] < ρ1[2]

    # The LDOS should increase outside the gap of a superconductor.
    assert ρ2[0] > ρ1[0]
    assert ρ2[3] > ρ1[3]

    # The lowest energy eigenvalue should increase in a superconductor.
    assert ε2 > ε1

def test_superconducting_gap_scaling():
    """The superconducting energy gap is given by its order parameter.

    We can test this by instantiating superconductors with varying
    order parameters, and then use the system's lowest energy
    eigenvalue as a measure of the energy gap in the spectrum.
    Ideally, these should be identical, but due to e.g. finite size
    effects in quick tests there may be small variations. We therefore
    settle for testing that one increases monotonically with the other.
    """
    # Construct a simple lattice.
    lattice = CubicLattice((32, 1, 1))
    system = Hamiltonian(lattice)

    # Define some representative order parameters.
    Δ_in = [0.0, 0.01, 0.03, 0.1, 0.3, 1.0]
    Δ_out = []

    # Construct the base Hamiltonian.
    t = 1.0
    μ = 1.5 * t
    with system as (H, Δ):
        for i in lattice.sites():
            H[i, i] = -μ * σ0
        for i, j in lattice.bonds():
            H[i, j] = -t * σ0

    # Perform the calculations.
    for Δ0 in Δ_in:
        # Update the order parameter.
        with system as (H, Δ):
            for i in lattice.sites():
                Δ[i, i] = Δ0 * jσ2

        # Diagonalize the Hamiltonian.
        eigvals, eigvecs = system.diagonalize()

        # The smallest positive eigenvalue is the gap.
        Δ_out.append(np.min(eigvals))

    # Check that the gaps increase with the order parameter.
    for Δ1, Δ2 in zip(Δ_out[:-1], Δ_out[1:]):
        assert Δ1 < Δ2

def test_magnetic_isotropy():
    """Test that the direction of a magnetic field is irrelevant.

    This is of course not true of all systems. But for a normal metal
    or singlet s-wave superconductor with a homogeneous magnetic field
    applied, the direction of that field should not affect any
    observables in the system. This can be tested by comparing the
    free energy and density of states for different field orientations.
    """
    lattice = CubicLattice((128, 1, 1))
    system = Hamiltonian(lattice)

    # Model parameters.
    i0 = (64, 0, 0)
    E0 = [0.0, 0.01]

    t = 1.0
    Δ0 = 0.1 * t
    M0 = 0.5 * Δ0

    T = 0.01 * t

    # Superconductor without a magnetic field.
    with system as (H, Δ):
        for i in lattice.sites():
            Δ[i,i] = -Δ0 * jσ2
        for i, j in lattice.bonds():
            H[i,j] = -t * σ0

    F0 = system.free_energy(T)
    ρ0 = system.ldos(i0, E0)[0]

    # Superconductor with random mangnetic fields.
    Fs = []
    ρs = []
    for i in range(10):
        θ = 2 * π * r()
        ϕ = 2 * π * r()
        σ = np.cos(θ) * σ1 + np.sin(θ) * np.cos(ϕ) * σ2 + np.sin(θ) * np.sin(ϕ) * σ3

        with system as (H, Δ):
            for i in lattice.sites():
                H[i,i] = -M0 * σ

        Fs.append(system.free_energy(T))
        ρs.append(system.ldos(i0, E0)[0])

    # Adding a magnetic field SHOULD change the observables.
    for F in Fs:
        assert not np.allclose(F0, F, rtol=1e-10)
    for ρ in ρs:
        assert not np.allclose(ρ0, ρ, rtol=1e-10)

    # Rotating that field SHOULD NOT change the observables.
    for F1, F2 in zip(Fs[:-1], Fs[1:]):
        assert np.allclose(F1, F2, rtol=1e-10)
    for ρ1, ρ2 in zip(ρs[:-1], ρs[1:]):
        assert np.allclose(ρ1, ρ2, rtol=1e-10)


def test_superconducting_spinvalve():
    """Test the (singlet) spin-valve effect in a superconductor.

    When two ferromagnets are attached to a superconductor, the free
    energy should be lowered when the two ferromagnets are aligned in
    an antiparallel manner since this avoids a net magnetic field
    penetrating the superconductor. This is the "spin-valve effect".
    """
    lattice = CubicLattice((128, 1, 1))
    system = Hamiltonian(lattice)

    # Functions to check which region we're in.
    def F1(i):
        """Left ferromagnet."""
        return i[0] < 32

    def F2(i):
        """Right ferromagnet."""
        return i[0] >= 128 - 32

    def S(i):
        """Superconductor."""
        return (not F1(i)) and (not F2(i))

    # System parameters.
    t = 1.0
    Δ0 = 0.3 * t
    M0 = 0.7 * t
    T = 0.001 * t

    # Case I: Parallel magnets.
    with system as (H, Δ):
        for i, j in lattice.bonds():
            H[i, j] = -t * σ0
        for i in lattice.sites():
            if S(i):
                Δ[i, i] = -Δ0 * jσ2
            if F1(i):
                H[i, i] = -M0 * σ3
            if F2(i):
                H[i, i] = -M0 * σ3

    E1 = system.free_energy(T)

    # Case II: Antiparallel magnets.
    # Note: Only F2 part changes here.
    with system as (H, Δ):
        for i in lattice.sites():
            if F2(i):
                H[i, i] = +M0 * σ3

    E2 = system.free_energy(T)

    assert E2 < E1


def test_odd_frequency():
    """Look for zero-energy peak formation in a superconductor.

    When an s-wave superconductor interacts with a ferromagnet, its
    Cooper pairs can be converted into "odd-frequency" spin-triplet
    pairs. In constrast to conventional Cooper pairs, these cause
    "zero-energy peaks" instead of "zero-energy gaps" to appear in the
    density of states. This code checks for such peak formation.
    """
    lattice = CubicLattice((128, 1, 1))
    system = Hamiltonian(lattice)

    # System parameters.
    t = 1.0
    Δ0 = 0.3 * t
    M0 = 0.5 * Δ0

    i0 = (63, 0, 0)
    E0 = [0.0, 0.05 * Δ0]

    # Case I: No magnet.
    with system as (H, Δ):
        for i, j in lattice.bonds():
            H[i, j] = -t * σ0
        for i in lattice.sites():
            Δ[i, i] = -Δ0 * jσ2

    Z1 = system.ldos(i0, E0)[0]

    # Case II: With magnet.
    with system as (H, Δ):
        for i in lattice.sites():
            H[i, i] = -M0 * σ2

    Z2 = system.ldos(i0, E0)[0]

    # Actual tests.
    assert Z1 >= 0
    assert Z2 >= Z1

def test_energy_temperature():
    """Test that the free energy decreases as a function of temperature.

    This can be seen from F = U - TS, where the entropy S tends to
    increase as a function of temperature in most simple systems.
    """
    # Instantiate a system with a magnetic domain wall as a test.
    lattice = CubicLattice((10, 10, 1))
    system = Hamiltonian(lattice)

    with system as (H, Δ):
        for i in lattice.sites():
            H[i, i] = -2.0 * σ0
        for i, j in lattice.bonds():
            H[i, j] = -1.0 * σ0

    # Calculate free energy at different temperatures.
    temperatures = [0.01, 0.1, 0.5, 1.0]
    free_energies = []
    for T in temperatures:
        F = system.free_energy(T)
        free_energies.append(F)

    # Compare the results.
    for T1, T2 in zip(free_energies[:-1], free_energies[1:]):
        assert T1 > T2
