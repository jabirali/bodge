# DESIGN
This note explains the various design decisions behind the current code.

## Language choice
Initially, I was hesitant about using Python for this project for performance reasons. However, after some initial experiments in Julia, I decided to switch to Python after all:

* I've mostly used Python for the past two years, and am therefore most familiar with the idioms, libraries, and tooling of that ecosystem. This is therefore a productive choice for me.
* In contrast to previous projects, tight-binding calculations mostly require operations on huge matrices, which libraries like `numpy` and `scipy` excel at. It's unlikely that the Python code is the bottleneck for this specific use case (cf. e.g. solving ODEs/PDEs).
* I have experience using Numba to speed up performance-sensitive code in Python if needed.

## Sparse matrices
To be able to solve systems with a large number of lattice points, it is essential to use *sparse matrices*, as implemented by e.g. the `scipy.sparse` library. I've decided to go for the `coo_matrix` format for construction and the `bsr_matrix` format for these reasons:

* Time to create a sparse BSR matrix for a 100x100x100 lattice:
	- COO matrix converted to BSR: 4.4 s.
	- LIL matrix converted to BSR: 14.4 s.
	- DOK matrix converted to BSR: 39.1 s.
* The row-ordered sparse matrices (CSR, BSR) are the two best options when performing a large number of matrix-vector multiplications, as occurs when working with a Chebyshev expansion.
* The Hamiltonian and Green function both consist of 4x4 dense submatrices due to the electron-hole and spin degrees of freedom at each site. Block formats (BSR) handles this.