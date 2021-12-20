"""Utility functions that are not directly related to the physics."""

import numba

from typing import Tuple


@numba.njit
def coord2index(inds: Tuple[int, int, int], dims: Tuple[int, int, int]):
    """Convert a set of indices (iˣ, iʸ, iᶻ) into a single index i.

    This is done in such a way that for a system with dimensions (Nˣ, Nʸ, Nᶻ),
    the indices map into a single index 0 ≤ i < Nˣ Nʸ Nᶻ in the "obvious" way.
    In order to retain compatibility with lower dimensions, specifying `inds`
    and `dims` with values like (iˣ, iʸ, 0) and (iˣ, 0, 0) is also supported. 
    """
    iˣ, iʸ, iᶻ = inds
    Nˣ, Nʸ, Nᶻ = dims

    return iˣ + iʸ * Nˣ + iᶻ * Nʸ * Nˣ


@numba.njit
def index2coord(ind: int, dims: Tuple[int, int, int]):
    """Convert a single index i into a set of indices (iˣ, iʸ, iᶻ).

    This function is by definition the inverse of `coord2index`, and the
    veracity of this is explicitly ensured via exhaustive unit testing.
    """
    Nˣ, Nʸ, Nᶻ = dims
    i = ind

    iˣ, i = i % Nˣ, i // Nˣ
    iʸ, i = i % Nʸ, i // Nʸ
    iᶻ = i

    return (iˣ, iʸ, iᶻ)
