# DESIGN
This note explains the various design decisions behind the current code.

## Language choice
Initially, I was hesitant about using Python for this project for performance reasons. However, after some initial experiments in Julia, I decided to switch to Python after all:

* I've mostly programmed in Python for the past years, and am therefore most familiar with the idioms, libraries, and tooling of the Python ecosystem. This is therefore a more productive choice for me.
* In contrast to my previous projects, tight-binding calculations mostly require operations on huge matrices, which is the specific use case that libraries like `numpy` excel at. It's unlikely that the Python code is the bottleneck in this specific case, since most of the heavy lifting is then relegated to C/C++/Fortran libraries.
* I do have experience using Numba to speed up performance-sensitive code in Python if that becomes necessary.

## Sparse matrices
To be able to solve systems with a large number of lattice point, it is essential to use *sparse matrices*, as implemented by e.g. the `scipy.sparse` library. I've decided to go for the `bsr_matrix` format for these reasons:

* Creation of 100x100x100 lattice:
	- COO matrix converted to BSR: 3.9 s.
	- LIL matrix converted to BSR: 14.4 s.
	- DOK matrix converted to BSR: 39.1 s.

* The Chebyshev expansion requires a large number of matrix-vector multiplications, which are operations that the row-ordered sparse matrices (`csr` and `bsr`) do well.
* The Hamiltonian and the Green function will likely both consist of 4x4 or 8x8 dense submatrices, due to the number of degrees of freedom associated with each lattice site. Block formats (like `bsr`) does this well.