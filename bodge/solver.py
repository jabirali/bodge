import os
import os.path
import sys
from multiprocessing import Pool

import scipy.sparse as sp
from h5py import File
from tqdm import tqdm

from .consts import *
from .hamiltonian import Hamiltonian
from .lattice import Lattice
from .stdio import *
from .typing import *


class Solution:
    """Interface to the results calculated by the `Solver` class.

    The spectral functions calculated by Bodge can be huge, and may not fit in
    computer memory. Moreover, it can be desirable to archive the results for
    reprocessing in the future. For these reasons, all the calculated results
    are stored in an HDF5 output file. This class wraps the generated HDF5
    file, such that the data can be accessed in a more seamless manner.
    """

    @typecheck
    def __init__(self, filename: str, lattice: Lattice) -> None:
        # Storage path for results.
        self.lattice: Lattice = lattice
        self.filename: str = filename

    @typecheck
    def integral(self) -> Sparse:
        """Accessor for the energy-integrated spectral function.

        `None` is returned if the quantity has not been calculated or stored.
        """
        with File(self.filename, "r") as file:
            if "integral" not in file:
                raise FileNotFoundError("Energy-integrated spectral function not found.")
            else:
                return unpack(file, "/integral")

    @typecheck
    def spectral(self) -> Iterator[Spectral]:
        """Accessor for the energy-resolved spectral function.

        `None` is returned if the quantity has not been calculated or stored.
        """
        with File(self.filename, "r") as file:
            if "spectral" not in file:
                raise FileNotFoundError("Energy-resolved spectral function not found.")
            else:
                ω = unpack(file, "/energies")
                for m in file["/spectral"]:
                    ω_m = ω[int(m)]
                    A_m = unpack(file, f"/spectral/{m}")

                    yield Spectral(ω_m, A_m)

    @typecheck
    def density(self) -> tuple[Array, Array]:
        """Calculate the local density of states."""
        ωs = []
        ds = []
        for ω, A in self.spectral():
            dof = A.blocksize[0]
            A_ii = A.diagonal()
            dos = np.real(A_ii[0::dof] + A_ii[1::dof])

            ωs.append(ω)
            ds.append(dos)

        return np.array(ωs), np.vstack(ds).T


class Solver:
    """User-facing interface for numerically calculating spectral functions.

    This part of the system is responsible for preparing relevant storage
    files and multiprocessing pools and then orchestrating calculations.
    Actual calculations are handled by `Kernel` and its derivatives.
    """

    @typecheck
    def __init__(
        self,
        kernel: Callable,
        hamiltonian: Hamiltonian,
        energies: int = 256,
        blocksize: int = 64,
        radius: Optional[int] = None,
        resolve: bool = False,
    ) -> None:
        # Save a reference to the Hamiltonian object.
        self.hamiltonian: Hamiltonian = hamiltonian

        # Number of energies to calculate the spectral function for.
        self.energies: int = energies

        # Linear scaling is achieved via a Local Krylov cutoff.
        self.radius: int
        if radius is not None:
            # Use the provided radius.
            self.radius = radius
            if self.radius < 1:
                raise RuntimeError("Subspace radius must be a positive integer.")
        else:
            # Empirically suficcient cutoff.
            self.radius = self.energies // 8

        # Parallelization is done by division into matrix blocks.
        self.blocksize: int = blocksize * hamiltonian.matrix.blocksize[0]
        self.blocks: int = self.hamiltonian.shape[1] // self.blocksize
        if self.blocksize * self.blocks != hamiltonian.shape[1]:
            raise RuntimeError(f"The blocksize should evenly divide the number of lattice sites.")

        # Whether to save the energy-resolved spectral function.
        self.resolve = resolve

        # Define a result file and instantiate the block solver.
        self.filename = "bodge.hdf"
        self.kernel = kernel(self.filename)

    @typecheck
    def __call__(self) -> Solution:
        # Load data from `self.filename`.
        log(self, "Preparing system for parallel calculations")
        with File(self.filename, "w") as file:
            pack(file, "/hamiltonian/matrix", self.hamiltonian.matrix)
            pack(file, "/hamiltonian/struct", self.hamiltonian.struct)
            pack(file, "/hamiltonian/scale", self.hamiltonian.scale)
            pack(file, "/numerics/blocks", self.blocks)
            pack(file, "/numerics/blocksize", self.blocksize)
            pack(file, "/numerics/energies", self.energies)
            pack(file, "/numerics/radius", self.radius)
            pack(file, "/numerics/resolve", self.resolve)

        # Parallel calculations with `multiprocessing`.
        log(self, "Calculating the spectral function in parallel")
        with Pool() as pool:
            try:
                block_names = pool.imap(self.kernel, range(self.blocks))
                block_names = [
                    *tqdm(
                        block_names,
                        total=self.blocks,
                        desc=" -> expanding",
                        unit="blk",
                        smoothing=0,
                    )
                ]
            except KeyboardInterrupt:
                print()
                log("Interrupt", "Exiting...")
                sys.exit(1)

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

        # Close up processed files.
        print(" -> cleaning up temporary files")
        for block_name in block_names:
            os.remove(block_name)

        # Return solution object.
        print(" -> done!\n")
        return Solution(self.filename, self.hamiltonian.lattice)


class Kernel:
    """Numerically calculate one block of a spectral function.

    This class is invoked indirectly by `Solver` to calculate the spectral
    function in parallel. In practice, you must implement the method `.solve`
    in a derived class, which is responsible for the actual calculations.
    """

    @typecheck
    def __init__(self, filename: str) -> None:
        # Filename to fetch input data from.
        self.filename: str = filename

    @typecheck
    def __call__(self, block: int) -> str:
        """Perform calculations at a given block index.

        This method first performs preparations that are useful for any
        algorithm for the polynomial expansion of the spectral function.
        Afterwards, it calls the `.solve` method to do the calculations.
        """
        try:
            # Output file for this block.
            base, ext = os.path.splitext(self.filename)
            self.blockname: str = f"{base}.{block:04d}{ext}"

            # Extract global parameters from data file.
            with File(self.filename, "r") as file:
                self.hamiltonian: Sparse = unpack(file, "/hamiltonian/matrix")
                self.skeleton: Sparse = unpack(file, "/hamiltonian/struct")
                self.scale = unpack(file, "/hamiltonian/scale")
                self.blocks: int = unpack(file, "/numerics/blocks")
                self.blocksize: int = unpack(file, "/numerics/blocksize")
                self.energies: int = unpack(file, "/numerics/energies")
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
            self.solve()

            # Return storage filename.
            return self.blockname
        except KeyboardInterrupt:
            os.remove(self.blockname)
            sys.exit(2)

    @typecheck
    def solve(self) -> None:
        """This method must be implemented by derived classes."""
        raise NotImplementedError
