# DESIGN

This note explains the various design decisions behind the current code.

## Language choice

Initially, I was hesitant about using Python for performance reasons. However,
after some initial experiments in Julia, I decided to go for Python after all:

* I've mostly used Python for the past two years, and am therefore most
  familiar with the idioms, libraries, and tooling of that ecosystem.
* In contrast to my previous numerical projects, tight-binding calculations
  mostly require operations on huge matrices, which libraries like `numpy`
  and `scipy` excel at. It's unlikely that the Python code will be the
  bottleneck for this specific use case (cf. e.g. solving ODEs/PDEs).
* I also have experience using Numba to speed up Python code if needed.

## Sparse matrices

To solve systems with a large number of lattice points, it is essential to
use *sparse matrices*, as implemented by e.g. the `scipy.sparse` library. I
went for `coo_matrix` for matrix construction and `bsr_matrix` for numerics:

* Time to create a sparse BSR matrix for a 100x100x100 lattice:
	- COO matrix converted to BSR: 4.4s.
	- LIL matrix converted to BSR: 14.4s.
	- DOK matrix converted to BSR: 39.1s.
* Row-ordered sparse matrices (CSR, BSR) are most efficient at matrix-vector
  multiplications, which are the bottleneck for our Chebyshev expansion.
* The Hamiltonian and Green function both consist of 4x4 dense submatrices due
  to the electron-hole and spin degrees of freedom. BSR accounts for this.
* In self-consistent calculations, we need to repeatedly update the
  Hamiltonian. The context manager is then easiest to write using a BSR
  matrix, since this lets us expose the blocks directly to the user.

For now, I have chosen `bsr_matrix` over `bsr_array`, even though this
interface is deprecated by `scipy`. The reason is that many useful functions
like `norm` and `eye` have not yet been ported to the new `array` interface.

## Blockwise expansion

The conceptually easiest approach is to generate the Chebyshev expansion for
the whole matrix simultaneously. However, as the intermediate matrices are
usually hundreds of times larger than the Hamiltonian itself, memory becomes
a problem for large systems. The other extreme is to process one unit vector
at a time instead of the whole identity matrix, as done by e.g. Nagai (2020).
However, using a high-level language like Python, the sparse matrix
construction cost becomes dominant, and the numerics slow to a crawl.

In this implementation, I've therefore implemented a blockwise expansion.
Blocksize of 1024 or 2048 appears empirically to be most efficient – on my
system, higher block sizes reduces CPU utilization (likely due to a cache or
memory bottleneck), while lower blocksizes become inefficient too (likely due
to more overhead being in Python rather than in the C/Fortran libraries).

## Parallelization

I've tried `joblib` with all backends and `multiprocessing`. The time used by
these two are the same with the `multiprocessing` backend, while the `loky`
backend is \~6% faster and the `threading` backend \~30% slower. However, no
`__main__` guard is required by `loky`, while `joblib` is easier to tune.

## Spectral radius

To use the Chebyshev expansion we need to compress the eigenvalue spectrum of
the Hamiltonian to (-1, +1). There are many bounds available for this
spectral radius, but a particularly efficient one is the 1-norm. In realistic
test cases on a 100x100x10 lattice, it takes only 0.2s to calculate this
quantity, yet the highest eigenvalue is usually 92-98% of this upper bound.

I also tried using `scipy.sparse.linalg.eigsh`, but it takes orders of
magnitude more time to finish, and even then provides unreliable results.
It could likely be tuned to do better, but the speed would be an issue.

## Gradual typing

I've decided to follow these guidelines:
- Types specified for all function arguments and
  return values to enable IDE/LSP autocompletion.
- Types specified for `self.*` variables in `__init__` when:
	1. The class is meant to be derived, in which case this enforces consistency;
	2. The members of the object may be directly accessed by other objects,
	   where this can be used to flag required refactoring after changes.

## Chebyshev approach

We have decided to perform a Chebyshev expansion of the Fermi operator F = f(H)
instead of the spectral function A(ω) or Green function G(ω). This follows the
PhD thesis by Benfenati. The reasoning is that it's easier to understand how
to incorporate finite temperatures in this scheme compared to the others.