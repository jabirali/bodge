"""
Bodge: Python package for efficient tight-binding modeling of superconductors.

For more information, please consult the bundled documentation and examples.
"""

from .common import *
from .hamiltonian import *
from .lattice import *

__author__ = "Jabir Ali Ouassou"
__version__ = "1.1.0"
__all__ = [
    # Core library
    "Lattice",
    "CubicLattice",
    "Hamiltonian",
    "Coord",
    "Coords",
    "Index",
    "Indices",
    # Helper functions
    "ssd",
    "swave",
    "pwave",
    "dwave",
    # Useful constants
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
    # ASCII alternatives
    "pi",
    "sigma",
    "sigma0",
    "sigma1",
    "sigma2",
    "sigma3",
    "jsigma",
    "jsigma0",
    "jsigma1",
    "jsigma2",
    "jsigma3",
]
