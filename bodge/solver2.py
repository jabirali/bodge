from multiprocessing import Pool

from h5py import File, Group
from rich import print
from tqdm import trange

from .consts import *
from .lattice import Lattice
from .physics import Hamiltonian
from .storage import *
from .typing import *


class BlockSolver:
    def __init__(self, filename: str):
        self.filename = filename

    def __call__(self, block):
        with File(self.filename, "r") as file:
            H = unpack(file, "/hamiltonian/matrix")

        return H


class Solver:
    """API for numerically calculating spectral functions and their integrals."""

    def __init__(
        self,
        hamiltonian: Hamiltonian,
        block_solver: BlockSolver,
        blocksize: int = 1024,
        radius: int = 4,
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

        # Define a result file and instantiate the block solver.
        self.filename = "bodge.hdf"
        self.block_solver = block_solver(self.filename)

    def __call__(self):
        print("[green]:: Preparing system for parallel calculations[/green]")
        with File(self.filename, "w") as file:
            pack(file, "/hamiltonian/matrix", self.hamiltonian.matrix)
            pack(file, "/hamiltonian/struct", self.hamiltonian.struct)

        print("[green]:: Calculating the spectral function in parallel[/green]")
        with Pool() as pool:

            print([x for x in pool.imap(self.block_solver, range(8))])
