#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from icecream import ic
from typer import run

np.set_printoptions(formatter={"float": "{: 0.12f}".format})


def free_energy(df: pd.DataFrame, s1: int, s2: int):
    """Calculate the free energy for a given configuration of two impurity spins.

    The spins are assumed to be specified as {±1, ±2, ±3} for the three cardinal
    axes, which is translated to the notations {x±, y±, z±} used in data files.
    If data is missing for any of the spin configurations, we just return NaN.
    """
    spins = {
        +1: "x+",
        +2: "y+",
        +3: "z+",
        -1: "x-",
        -2: "y-",
        -3: "z-",
    }

    s1_ = spins[s1]
    s2_ = spins[s2]

    try:
        return float(df[(df["s1"] == s1_) & (df["s2"] == s2_)].E)
    except:
        return np.nan


def effective_energy(df: pd.DataFrame):
    """Calculate the couplings in the effective free energy for the two spins.

    This includes a Heisenberg-like J, Dzyaloshinskii-Moriya-like D, preferred
    spin orientation μ for each impurity, and the spin-averaged free energy ε0.
    """
    results = []
    for (d, δ), df_ in df.groupby(["dvec", "sep"]):
        ic(δ)

        # Calculate spin-independent free energy (rank 0).
        ε0 = 0.0
        for s1 in [+1, +2, +3, -1, -2, -3]:
            for s2 in [+1, +2, +3, -1, -2, -3]:
                ε0 += free_energy(df_, s1, s2) / 36

        ic(ε0)

        # Calculate the preferred orientation of each spin (rank 1).
        μ1 = np.zeros(3)
        μ2 = np.zeros(3)

        for i1, s1 in enumerate([1, 2, 3]):
            for s2 in [+1, +2, +3, -1, -2, -3]:
                μ1[i1] += free_energy(df_, +s1, s2) / 12
                μ1[i1] -= free_energy(df_, -s1, s2) / 12
        for i2, s2 in enumerate([1, 2, 3]):
            for s1 in [+1, +2, +3, -1, -2, -3]:
                μ2[i2] += free_energy(df_, s1, +s2) / 12
                μ2[i2] -= free_energy(df_, s1, -s2) / 12

        ic(μ1)
        ic(μ2)

        # Calculate the interactions between the two spins (rank 2).
        η = np.zeros((3, 3))
        for i1, s1 in enumerate([1, 2, 3]):
            for i2, s2 in enumerate([1, 2, 3]):
                η[i1, i2] += free_energy(df_, +s1, +s2) / 4
                η[i1, i2] -= free_energy(df_, +s1, -s2) / 4
                η[i1, i2] -= free_energy(df_, -s1, +s2) / 4
                η[i1, i2] += free_energy(df_, -s1, -s2) / 4

        ic(η)

        # Calculate averaged orientational preference.
        μ = (μ1 + μ2) / 2
        ic(μ)

        # Calculate the Heisenberg-like interaction.
        J = np.zeros(3)
        for i in range(3):
            J[i] = η[i, i]

        ic(J)

        # Calculate the Dzyaloshinskii-Moriya-like interaction.
        D = np.zeros(3)
        for i in range(3):
            j = (i + 1) % 3
            k = (i + 2) % 3
            D[i] = (η[j, k] - η[k, j]) / 2

        ic(D)

        # Save the effective couplings in the free energy.
        results.append(
            {
                "d": d,
                "δ": δ,
                "ε0": ε0,
                "μx": μ[0],
                "μy": μ[1],
                "μz": μ[2],
                "Jx": J[0],
                "Jy": J[1],
                "Jz": J[2],
                "Dx": D[0],
                "Dy": D[1],
                "Dz": D[2],
            }
        )

    # Construct a dataframe out of the results.
    return pd.DataFrame(results)


def plot(df: pd.DataFrame, var: str):
    """Plot a particular effective energy coupling as function of separation."""
    _, ax = plt.subplots()
    sns.lineplot(
        data=df,
        ax=ax,
        x="δ",
        y=var,
        hue="d",
    )
    ax.legend(title="d-vector")
    return ax


def main(filename: str):
    """Plot the results generated by RKKY calculations."""

    # Load simulation results.
    df = pd.read_csv(filename, names=["dvec", "s1", "s2", "sep", "E"], skipinitialspace=True)
    df = df.sort_values(by=["dvec", "sep", "s1", "s2"])
    ic(df)

    # Postprocess results.
    df = effective_energy(df)
    ic(df)

    # Visualize the results.
    ax = plot(df, "ε0")
    ax.set_xlabel(r"Distance $δ/a$")
    ax.set_ylabel(rf"Spin-independent energy $E_0/t$")

    for axis in ["x", "y", "z"]:
        ax = plot(df, f"μ{axis}")
        ax.set_xlabel(r"Distance $δ/a$")
        ax.set_ylabel(rf"Magnetic moment $μ_{axis}/t$")

    for axis in ["x", "y", "z"]:
        ax = plot(df, f"J{axis}")
        ax.set_xlabel(r"Distance $δ/a$")
        ax.set_ylabel(rf"Heisenberg-like coupling $J_{axis}/t$")

    for axis in ["x", "y", "z"]:
        ax = plot(df, f"D{axis}")
        ax.set_xlabel(r"Distance $δ/a$")
        ax.set_ylabel(rf"DMI-like coupling $D_{axis}/t$")

    plt.show()


if __name__ == "__main__":
    run(main)
