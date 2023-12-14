"""Bodge: Python package for efficient tight-binding modeling of superconductors.

This package is used to construct a "BOgoliubov-DeGEnnes Hamiltonian", which
is a tight-binding model that describes superconductors. The package is quite
general and can be used to describe non-superconducting systems as well; one
can easily model e.g. magnetism, spin-orbit coupling, and random impurities.
It also bundles some useful utility functions that operate on the resulting
matrices, e.g. for calculating the density of states in the system.

Some design principles that underlie this software package are:

1. __Intuitiveness__: The Python code should be as similar to the "pen and paper"
   model as possible. This reduces the time needed to study a new physical system,
   and reduces the risk of bugs due to deviations between the code and the paper.
   In practice, this is accomplished using a "context manager", which converts a
   somewhat human-readable system description into more precise matrix operations.
2. __Sparse matrices__: Materials with only on-site and nearest-neighbor
   interactions end up with extremely sparse Hamiltonian matrices. Using a
   computer model that acknowledges this permits the modeling of much larger
   systems than what would otherwise be computationally feasible. However,
   constructing and indexing sparse matrices can be a chore. This package
   provides an intuitive high-level interface that abstracts away this task.
3. __Real-space modeling__: Many similar packages employ a momentum-space
   description of physical systems. However, phenomena such as interfaces
   or impurity-bound states can be more naturally described in real space.
   This package therefore focuses on tight-binding models in real space.
4. __Symmetries__: Typically, the Hamiltonian matrix needs to respect certain
   symmetries, such as the electron-hole symmetry of superconductors and the
   Hermitian symmetry of the final matrix. Manually constructing the Hamiltonian
   matrix is not only tedious but also a potential source of user errors. This
   package avoids that by auto-filling parts of the matrix from symmetry.
5. __Correctness__: We aim towards 100% test coverage to prevent bugs and
   regressions in the code base. Moreover, the code uses type hints with
   runtime type checking, which eliminates another common source of errors. 
   There is also a runtime check that ensures the Hamiltonian is hermitian.
6. __Flexibility__: To the extent possible, it should be possible to reuse the
   `bodge` classes for different purposes. For instance, the `Lattice` that the
   `Hamiltonian` object is constructed on is completely general. This means that
   if a user wants to define a Hamiltonian matrix on e.g. a honeycomb lattice or
   a quasicrystal, they just have to create a class that derives from `Lattice`,
   and implement some iteration logic for how to traverse that custom lattice.

For more information, please consult the documentation and examples.
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
]
