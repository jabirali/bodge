import numpy as np

# Fundamental constants.
π = np.pi

# Pauli matrices used to represent spin.
σ0 = np.array([[+1, 0], [0, +1]], dtype=np.complex64)
σ1 = np.array([[0, +1], [+1, 0]], dtype=np.complex64)
σ2 = np.array([[0, -1j], [+1j, 0]], dtype=np.complex64)
σ3 = np.array([[+1, 0], [0, -1]], dtype=np.complex64)

# Compact notation for imaginary versions.
jσ0 = 1j * σ0
jσ1 = 1j * σ1
jσ2 = 1j * σ2
jσ3 = 1j * σ3

