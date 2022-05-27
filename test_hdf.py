#!/usr/bin/env python3

import numpy as np
from h5py import File

# Define some variables.
a = True
b = 3.14
c = np.array([1, 2, 3])

# Save the variables to `test.hdf`.
with File("test.hdf", "w") as file:
    file["/results/A"] = a
    file["/results/B"] = b
    file["/results/C"] = c

# Load the variables again.
with File("test.hdf", "r") as file:
    A = file["/results/A"][...]
    B = file["/results/B"][...]
    C = file["/results/C"][...]

    print(A, B, C)
