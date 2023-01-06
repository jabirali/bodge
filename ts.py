#!/usr/bin/env python

"""Calculate the local density of states around a magnetic impurity.

This is useful to e.g. determine the YSR bound states that might exist
in such materials, which is likely related to RKKY oscillations.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from bodge import *
from bodge.utils import ldos, pwave

# Construct a 2D lattice.
Lx = 80
Ly = 80
Lz = 1

lattice = CubicLattice((Lx, Ly, 1))

# %%
plt.plot([1, 2, 3], [4, 5, 6])
