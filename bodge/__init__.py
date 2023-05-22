"""Sparse solver for large-scale tight-binding systems."""

from .chebyshev import *
from .common import *
from .fermi import *
from .hamiltonian import *
from .lattice import *
from .utils import *

__author__ = "Jabir Ali Ouassou"
__version__ = "0.2.0"
__all__ = [
    "CubicLattice",
    "FermiMatrix",
    "Hamiltonian",
    "Coord",
    "Coords",
    "Index",
    "Indices",
    "deform",
    "critical_temperature",
    "critical_temperature_2",
    "critical_temperature_3",
    "critical_temperature_4",
    "diagonalize",
    "dwave",
    "free_energy",
    "ldos",
    "pwave",
    "spectral",
    "swave",
    "π",
    "σ",
    "σ0",
    "σ1",
    "σ2",
    "σ3",
    "jσ",
    "jσ0",
    "jσ1",
    "jσ2",
    "jσ3",
]
