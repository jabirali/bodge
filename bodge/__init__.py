import os
import sys
from glob import glob
from signal import SIGINT, signal

from .chebyshev import *
from .consts import *
from .lattice import *
from .physics import *
from .solver import *

__author__ = "Jabir Ali Ouassou"
__version__ = "0.0.4"
__all__ = [
    "CubicLattice",
    "Hamiltonian",
    "Solver",
    "chebyshev",
    "σ0",
    "σ1",
    "σ2",
    "σ3",
    "jσ0",
    "jσ1",
    "jσ2",
    "jσ3",
]

# Exit gracefully upon Ctrl-C.
def interrupt(sig, frame):
    print()
    log("SIGINT", "Cleaning up...")
    for file in glob("./bodge.*.hdf"):
        os.remove(file)
    log("SIGINT", "Exiting...")
    sys.exit(1)


signal(SIGINT, interrupt)
