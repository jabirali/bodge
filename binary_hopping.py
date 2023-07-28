#!/usr/bin/env python3

"""Binary search in the inter-layer hopping parameter of an S/FM system."""

import logging as log
import sys

import numpy as np
import typer

from bodge import *


def main(tau: float, delta: str):
    # Construct an appropriate lattice, including functions to determine
    # whether a particular region is superconducting or normal metallic.
    Lx, Ly, Lz = 64, 64, 2
    lattice = CubicLattice((Lx, Ly, Lz))
    log.debug(f"Lattice: {lattice}")

    def NM(i: Coord):
        x, y, z = i
        return z == 1

    def SC(i: Coord):
        x, y, z = i
        return z == 0 and x >= Lx // 4 and x < 3 * Lx // 4 and y >= Ly // 4 and y < 3 * Ly // 4

    def IN(i: Coord):
        return NM(i) or SC(i)

    # Load the interpolated profiles.
    filename = f"m_{delta}.npz"
    log.debug(f"Filename: {filename}")
    with np.load(filename) as f:
        mx, my, mz = f["mx"], f["my"], f["mz"]


if __name__ == "__main__":
    log.basicConfig(
        filename="main.log",
        filemode="w",
        format="%(asctime)s %(levelname)s \t%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=log.DEBUG,
    )
    log.info(f"{' '.join(sys.argv)}")
    typer.run(main)
