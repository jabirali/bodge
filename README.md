# Bodge

Bodge is a Python package for constructing large real-space tight-binding
models. Although quite general tight-binding models can be constructed, we
focus on the Bogoliubov-DeGennes ("BoDGe") Hamiltonian, which is used to model
superconductivity in clean materials. In other words: If you want a lattice
model for superconducting nanostructures, and want something that is
computationally efficient yet easy to use, you've come to the right place.

Internally, the package uses a sparse matrix (`scipy.sparse`) to represent the
Hamiltonian, which allows you to efficiently construct tight-binding models
with millions of lattice sites if needed. However, you can also easily convert
the result to a dense matrix (`numpy.array`) if that's more convenient. The
package follows modern software development practices: full test coverage
(`pytest`), runtime type checks (`beartype`), and PEP-8 compliance (`black`).

## Quickstart

One of the main features of this package is a simple syntax if all you want to
do is to create a real-space lattice Hamiltonian with some superconductivity.
For instance, say that you want a $100\times100$ s-wave superconductor with
a chemical potential $μ = -3t$, s-wave superconducting gap $Δ_s = 0.1t$,
magnetic spin splitting along the z-direction $m = 0.05t$, and with
nearest-neighbor hopping parameter $t = 1$. Using Bodge, you write:

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
intentionally very close to how one might write the corresponding equations
with pen and paper. It is similarly easy to model more complex systems: you can
e.g. easily add p-wave or d-wave superconductivity, antiferromafgnetism,
spin-orbit coupling, altermagnetism, or inhomogeneities like interfaces. In
general, `H[i, i]` and `H[i, j]` set the on-site and nearest-neighbor "normal"
terms in the Hamiltonian matrix, and `Δ[i, i]` and `Δ[i, j]` set the
corresponding pairing terms representing superconductivity. Once you're done,
you can e.g. call `system("dense")` to get a NumPy array, `system("csr")` to
get a Compressed Sparse Row matrix, or `system.diagonalize()` to get the
eigenvectors and eigenvalues of the Hamiltonian you just made.

The syntax used to construct the Hamiltonian is designed to look like array
operations, but this is just a friendly interface; under the hood a lot of
"magic" occurs to efficiently translate what you see into sparse matrix
operations, while enforcing particle-hole and Hermitian symmetries.

## Installation

To use the default Python version on your system, simply run:

	make install

This will create a virtual environment in a subfolder called `venv`,
and then install Bodge into that virtual environment. If you prefer to
use a newer Python version (recommended), first install this via your
package manager of choice. For example:

	brew install python@3.11          # MacOS with HomeBrew
	sudo apt install python3.11-full  # Ubuntu GNU/Linux

Afterwards, mention what Python version to use when installing Bodge:

	make install PYTHON=python3.11

Run `make` without any command-line arguments to see how to proceed
further. This should provide information on how to run the bundled
unit tests, run scripts that use the Bodge package, or run the
autoformatter after updates.

## Acknowledgements

I wrote most of this code as a PostDoc in the research group of Prof. Jacob
Linder at the [Center for Quantum Spintronics, NTNU, Norway][1]. I would like
to thank Jacob for introducing me to the BdG formalism that is implemented in
this package – and before that, to the theory superconductivity in general.
During my PostDoc, this code was used to investigate several interesting
research topics in superconductivity, including the [dc Josephson effect in
altermagnets][2] and [RKKY interaction in triplet superconductors][3].

[1]: https://www.ntnu.edu/quspin
[2]: https://doi.org/10.1103/PhysRevLett.131.076003
[3]: https://dx.doi.org/10.1103/PhysRevB.109.174506

