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
    "Hamiltonian",
    "FermiMatrix",
    "diagonalize",
    "pwave",
    "dwave",
    "free_energy",
    "spectral",
    "deform",
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
