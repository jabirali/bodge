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
	- COO matrix converted to BSR: 4.4s.
	- LIL matrix converted to BSR: 14.4s.
	- DOK matrix converted to BSR: 39.1s.
* The row-ordered sparse matrices (CSR, BSR) are the two best options when performing a large number of matrix-vector multiplications, as occurs when working with a Chebyshev expansion.
* The Hamiltonian and Green function both consist of 4x4 dense submatrices due to the electron-hole and spin degrees of freedom at each site. Block formats (BSR) handles this.
* In self-consistent calculations, we need to repeatedly update the Hamiltonian. It's therefore best to use the BSR format and not the COO format in the context manager.

## Spectral radius
To use the Chebyshev expansion of the Green function, we need to compress the eigenvalue spectrum of the Hamiltonian to (-1, +1). There are many bounds available for this *spectral radius*, but a particularly efficient one turns out to be the 1-norm of the matrix. In realistic test cases on a 100x100x10 lattice, it takes only 0.2s to calculate this quantity, yet the highest eigenvalue is in practice 92-98% of this upper bound.

I also tried using `scipy.sparse.linalg.eigsh`, but it takes orders of magnitude more time to finish, and even then provides unreliable results. It could likely be tuned to do better, but the speed would be an issue.