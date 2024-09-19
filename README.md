# Bodge

[![JOSS](https://joss.theoj.org/papers/18b48f694511e8c02a6b56375855fd0c/status.svg)](https://joss.theoj.org/papers/18b48f694511e8c02a6b56375855fd0c)
[![PyPI](https://img.shields.io/pypi/v/bodge?logo=python&logoColor=white&label=PyPI)](https://pypi.org/project/bodge/)
[![Docs](https://img.shields.io/badge/Docs-tutorial-blue?logo=readme&logoColor=white)](https://jabirali.github.io/bodge/)
[![Tests](https://github.com/jabirali/bodge/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/jabirali/bodge/actions/workflows/unit-tests.yml)

Bodge is a Python package for constructing large real-space tight-binding
models. Although quite general tight-binding models can be constructed, we focus
on the Bogoliubov-DeGennes ("BoDGe") Hamiltonian, which is used to model
superconductivity in clean materials. In other words: If you want a lattice
model for superconducting nanostructures, and want something that is
computationally efficient yet easy to use, you've come to the right place.

Research papers where this code has been used include:
- [DC Josephson effect in altermagnets][2]
- [RKKY interaction in triplet superconductors][3]
- [Dzyaloshinskii-Moriya spin-spin interaction from mixed-parity superconductivity][6]

Bodge can be used on anything from a normal laptop to an HPC cluster, as long as
it runs an operating system where the SciPy stack is available (e.g. Linux,
MacOS, or Windows). It is mainly meant for CPU-based calculations on one node
(i.e. no MPI support), however some functions on the `development` branch
contains optional support for GPU-based calculations via CuPy (CUDA).

Internally, Bodge uses a sparse matrix (`scipy.sparse`) to represent the Hamiltonian,
which allows you to efficiently construct tight-binding models with millions of
lattice sites if needed. However, you can also easily convert the result to a
dense matrix (`numpy.array`) if that's more convenient. The package follows
modern software development practices: near-complete test coverage (`pytest`),
fast runtime type checking (`beartype`), and mostly PEP-8 compliant (`black`).

The full documentation of the Bodge package is available [here][0].

## Quickstart

This package is [published on PyPi][4], and is easily installed via `pip`:

    pip install bodge

Bodge should be quite easy to use if all you want is a real-space lattice
Hamiltonian with superconductivity. For instance, consider a $100a\times100a$
s-wave superconductor with a chemical potential $μ = -3t$, superconducting gap
$Δ_s = 0.1t$, magnetic spin splitting along the $z$ axis $m = 0.05t$, and
nearest-neighbor hopping $t = 1$. Using Bodge, you can just write:

```python
from bodge import *

lattice = CubicLattice((100, 100, 1))
system = Hamiltonian(lattice)

t = 1
μ = -3 * t
m = 0.05 * t
Δs = 0.10 * t

with system as (H, Δ):
    for i in lattice.sites():
        H[i, i] = -μ * σ0 -m * σ3
        Δ[i, i] = -Δs * jσ2
    for i, j in lattice.bonds():
        H[i, j] = -t * σ0
```

If you are familiar with tight-binding models, you might notice that this is
intentionally very close to how one might write the corresponding equations with
pen and paper. It is similarly easy to model more complex systems: you can
e.g. easily add p-wave or d-wave superconductivity, magnetism, altermagnetism,
antiferromagnetism, spin-orbit coupling, etc. The lattice site indices are
implemented as tuples `i = (x_i, y_i, z_i)`, so that you can easily use the
lattice coordinates to design inhomogeneous materials via e.g. `if`-tests.

The syntax used to construct the Hamiltonian is designed to look like array
operations, but this is just a friendly interface; under the hood, some "magic"
is required to efficiently translate what you see above into sparse matrix
operations, while enforcing particle-hole and Hermitian symmetries. Once you're
done with the construction, you can call `system.matrix()` to extract the
Hamiltonian matrix itself (in dense or sparse form), or use methods such as
`system.diagonalize()` and `system.free_energy()` to get derived properties.

Formally, the Hamiltonian operator that corresponds to the constructed matrix is

$$\mathcal{H} = E_0 + \frac{1}{2} \sum_{ij} \hat{c}^\dagger_i \hat{H}_{ij} \hat{c}_j,$$

where
$\hat{c}\_i = (c_{i\uparrow}, c_{i\downarrow}, c_{i\uparrow}^\dagger, c_{i\downarrow}^\dagger)$
is a vector of all spin-dependent electron operators on lattice site $i$ and
$E_0$ is a constant. The $4\times4$ matrix $\hat{H}_{ij}$ in Nambu⊗Spin space is
generally further decomposed into $2\times2$ blocks in spin space:

$$\hat{H}\_{ij} = \begin{pmatrix} H_{ij} & \Delta_{ij} \\\\ \Delta^\dagger_{ij} & -H^*_{ij} \end{pmatrix}$$

It is precisely these $2\times2$ blocks $H_{ij}$ and $\Delta_{ij}$ that are
specified when you provide Bodge with values for `H[i, j]` and `∆[i, j]`. After
the matrix construction completes, you obtain a $4N\times4N$ matrix in
Lattice⊗Nambu⊗Spin space, where $N$ is the number of lattice sites. You don't
need to specify the bottom row of $\hat{H}_{ij}$ as these follow from symmetry,
and Bodge will warn you if the Hamiltonian is non-Hermitian.

For more information on usage, please see [the full documentation][0].

## Development

After cloning the [Git repository][5] on a Unix-like system, you can run:

	make install

This will create a virtual environment in a subfolder called `venv`, and then
install Bodge into that virtual environment. If you prefer to use a newer Python
version (recommended), first install this via your package manager of choice.

For example:

	brew install python@3.11          # MacOS with HomeBrew
	sudo apt install python3.11-full  # Ubuntu GNU/Linux

Afterwards, mention what Python version to use when installing Bodge:

	make install PYTHON=python3.11

Run `make` without any command-line arguments to see how to proceed
further. This should provide information on how to run the bundled
unit tests, run scripts that use the Bodge package, or run the
autoformatter after you have updated the code. PRs are welcome!

## Acknowledgements

I wrote most of this code as a PostDoc in the research group of Prof. Jacob
Linder at the [Center for Quantum Spintronics, NTNU, Norway][1]. I would like to
thank Jacob for introducing me to the BdG formalism that is implemented in this
package – and before that, to the theory superconductivity in general.

[0]: https://jabirali.github.io/bodge/
[1]: https://www.ntnu.edu/quspin
[2]: https://doi.org/10.1103/PhysRevLett.131.076003
[3]: https://doi.org/10.1103/PhysRevB.109.174506
[4]: https://pypi.org/project/bodge/
[5]: https://github.com/jabirali/bodge
[6]: https://doi.org/10.48550/arXiv.2407.07144
