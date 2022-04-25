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

For now, I have chosen to use `bsr_matrix` over `bsr_array`, even though this interface is deprecated by `scipy`. The reason is simply that many useful functions like `norm` and `eye` have not yet been ported to use the new `array` interface.

## Blockwise expansion
When it comes to the Chebyshev expansion, the conceptually easiest approach is to generate the Chebyshev expansion for the whole matrix simultaneously. However, as the intermediate matrices are usually hundreds of times larger than the Hamiltonian, memory becomes a problem for large systems. The other extreme is to process one unit vector at a time instead of the whole identity matrix, as done by e.g. Nagai (2020). However, using a high-level language like Python, the sparse matrix construction cost becomes dominant, and the numerics slow to a crawl.

In this implementation, I've therefore implemented a blockwise expansion. Blocksize of 1024 or 2048 appears empirically to be most efficient – on my system, higher block sizes reduces CPU utilization (likely due to a cache or memory bottleneck), while lower blocksizes also becomes inefficient (likely due to more overhead being in Python rather than in C/Fortran libraries).

## Parallelization
I've tried `joblib` with all backends and `multiprocessing` library. The time used by these two are the same with the `multiprocessing` backend, while the `loky` backend is \~6% faster and the `threading` backend is \~30% slower. However, no `__main__` guard is required by `loky`, and `joblib` is perhaps easier to tune.

## Spectral radius
To use the Chebyshev expansion of the Green function, we need to compress the eigenvalue spectrum of the Hamiltonian to (-1, +1). There are many bounds available for this *spectral radius*, but a particularly efficient one turns out to be the 1-norm of the matrix. In realistic test cases on a 100x100x10 lattice, it takes only 0.2s to calculate this quantity, yet the highest eigenvalue is in practice 92-98% of this upper bound.

I also tried using `scipy.sparse.linalg.eigsh`, but it takes orders of magnitude more time to finish, and even then provides unreliable results. It could likely be tuned to do better, but the speed would be an issue.

## Gradual typing
I've decided to follow these guidelines:
- Types specified for all function arguments and return values to enable IDE/LSP autocompletion.
- Types specified for the `self.*` variables defined in `__init__` when:
	1. The class is meant to be derived, in which case this enforces consistency;
	2. The members of the object may be directly accessed by other objects, where this can be used to flag required refactoring after changes.
