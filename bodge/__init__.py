"""
Bodge: Python package for efficient tight-binding modeling of superconductors.

For more information, please consult the bundled documentation and examples.
"""

from .common import *
from .hamiltonian import *
from .lattice import *
from .utils import *

__author__ = "Jabir Ali Ouassou"
__version__ = "0.9.0"
__all__ = [
    "Lattice",
    "CubicLattice",
    "Hamiltonian",
    "Coord",
    "Coords",
    "Index",
    "Indices",
    "diagonalize",
    "free_energy",
    "spectral",
    "ldos",
    "deform",
    "swave",
    "pwave",
    "dwave",
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
    "sigma0",
    "sigma1",
    "sigma2",
    "sigma3",
]
