#!/usr/bin/env python

"""Revised exploration of superconductor-altermagnet junctions.

Based on some initial promising results in `altermagnets.ipynb`, we now attempt
to further explore the physics of 0-π oscillations in Josephson junctions with
altermagnetic interlayers -- using e.g. more realistic material parameters.
"""

# %% Common imports
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm, trange

from bodge import *

# %%
plt.plot([1,2,3], [1,4,9])

