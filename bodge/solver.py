import os
import os.path
from multiprocessing import Pool

import scipy.sparse as sp
from h5py import File, Group
from rich import print
from tqdm import tqdm, trange

from .consts import *
from .lattice import Lattice
from .physics import Hamiltonian
from .storage import *
from .typing import *


class BlockSolver:
    """Numerically calculate one block of a spectral function.

    This class is invoked indirectly by `Solver` to calculate the spectral
    function in parallel. In practice, you must implement the method `.solve`
    in a derived class, which is responsible for the actual calculations.
    """

    def __init__(self, filename: str) -> None:
        # Filename to fetch input data from.
        self.filename: str = filename

    def __call__(self, block: int) -> str:
        """Perform calculations at a given block index.

        This method first performs preparations that are useful for any
        algorithm for the polynomial expansion of the spectral function.
        Afterwards, it calls the `.solve` method to do the calculations.
        """
        # Output file for this block.
        base, ext = os.path.splitext(self.filename)
        self.blockname: str = f"{base}.{block:04d}{ext}"

        # Extract global parameters from data file.
        with File(self.filename, "r") as file:
            self.hamiltonian: Sparse = unpack(file, "/hamiltonian/matrix")
            self.skeleton: Sparse = unpack(file, "/hamiltonian/struct")
            self.blocksize: int = unpack(file, "/numerics/blocksize")
            self.blocks: int = unpack(file, "/numerics/blocks")
            self.radius: int = unpack(file, "/numerics/radius")
            self.resolve: bool = unpack(file, "/numerics/resolve")

        # Instantiate the current block of the identity matrix.
        diag = np.repeat(np.int8(1), self.blocksize)
        offset = -block * self.blocksize
        shape = (self.hamiltonian.shape[0], self.blocksize)
        identity = sp.dia_matrix((diag, [offset]), shape, dtype=np.int8)

        self.block_identity = identity.tobsr(self.hamiltonian.blocksize)

        # Projection with this mask retains only local terms (up to nearest
        # neighbors), which are the relevant terms in the spectral function.
        self.block_neighbors = self.skeleton @ self.block_identity

        # Projection with this mask retains all terms within a bubble of
        # a given radius. This defines the Local Krylov subspace used for
        # intermediate calculations in the Green function expansions.
        mask = self.block_neighbors
        for _ in range(self.radius - 1):
            mask = self.skeleton @ mask
        mask.data[...] = 1

        self.block_subspace = Sparse(mask, dtype=np.int8)

        # Run actual calculations.
        self.solve(block)

        # Return storage filename.
        return self.blockname

    def solve(self, block: int) -> None:
        """This method must be implemented by derived classes."""
        raise NotImplementedError


class Solver:
    """User-facing interface for numerically calculating spectral functions.

    This part of the system is responsible for preparing relevant storage
    files and multiprocessing pools and then orchestrating calculations.
    Actual calculations are handled by `BlockSolver` and its derivatives.
    """

    def __init__(
        self,
        block_solver: Callable,
        hamiltonian: Hamiltonian,
        blocksize: int = 1024,
        radius: int = 4,
        resolve: bool = False,
    ):
        # Save a reference to the Hamiltonian object.
        self.hamiltonian: Hamiltonian = hamiltonian

        # Linear scaling is achieved via a Local Krylov cutoff.
        self.radius: int = radius
        if self.radius < 1:
            raise RuntimeError("Krylov cutoff radius must be a positive integer.")

        # Parallelization is done by division into matrix blocks.
        self.blocksize: int = blocksize
        self.blocks: int = self.hamiltonian.shape[1] // blocksize
        if self.blocksize * self.blocks != hamiltonian.shape[1]:
            raise RuntimeError(f"Hamiltonian shape must be a multiple of {blocksize}.")

        # Whether to save the energy-resolved spectral function.
        self.resolve = resolve 

        # Define a result file and instantiate the block solver.
        self.filename = "bodge.hdf"
        self.block_solver = block_solver(self.filename)

    def __call__(self):
        # Load data from `self.filename`.
        print("[green]:: Preparing system for parallel calculations[/green]")
        with File(self.filename, "w") as file:
            pack(file, "/hamiltonian/matrix", self.hamiltonian.matrix)
            pack(file, "/hamiltonian/struct", self.hamiltonian.struct)
            pack(file, "/numerics/blocksize", self.blocksize)
            pack(file, "/numerics/blocks", self.blocks)
            pack(file, "/numerics/radius", self.radius)
            pack(file, "/numerics/resolve", self.resolve)

        # Parallel calculations with `multiprocessing`.
        print("[green]:: Calculating the spectral function in parallel[/green]")
        with Pool() as pool:
            block_names = pool.imap(self.block_solver, range(self.blocks))
            block_names = [*tqdm(block_names, total=self.blocks, desc=" -> expanding", unit="blk")]

        # Determine which outputs were calculated.
        matrices = []
        others = []
        with File(block_names[0], "r") as file:
            # Energy-integrated spectral function.
            if "integral" in file:
                matrices.append("/integral")
            # Energy-resolved spectral function.
            if "spectral" in file:
                for m in file["/spectral"]:
                    matrices.append(f"/spectral/{m}")
            # Miscellaneous parameters.
            for path in file:
                if path not in ["integral", "spectral"]:
                    others.append(path)

        # Merge the calculated blocks.
        with File(self.filename, "r+") as file:
            # Reconstruct all matrices from blocks.
            matrix_range = tqdm(matrices, desc=" -> merging", unit="mat")
            for matrix in matrix_range:
                # Extract all blocks for current matrix.
                blocks = []
                for block_name in block_names:
                    with File(block_name, "r") as block_file:
                        blocks.append(unpack(block_file, matrix))

                # Stack the blocks and store in output file.
                pack(file, matrix, sp.hstack(blocks, "bsr"))

            # Transfer miscellaneous outputs.
            with File(block_names[0], "r") as block_file:
                for other in others:
                    pack(file, other, unpack(block_file, other))

            print([f for f in file])

        # Close up processed files.
        for block_name in block_names:
            os.remove(block_name)
