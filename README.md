# Bodge

Bodge is a Python package for constructing large real-space tight-binding
models. Although quite general tight-binding models can be constructed, we
focus on the Bogoliubov-DeGennes ("BoDGe") Hamiltonian, which is used to model
superconductivity in clean materials. In other words: If you want a lattice
model for superconducting nanostructures, and want something that is easy to
use and has high performance, then you've come to the right place.

Internally, the package uses a sparse matrix (`scipy.sparse`) to represent the
Hamiltonian, which allows you to efficiently construct tight-binding models
with millions of lattice sites if needed. However, you can also easily convert
the result to a dense matrix (`numpy.array`) if that's more convenient. The
package follows modern software development practices: full test coverage
(`pytest`), runtime type checks (`beartype`), and PEP-8 compliance (`black`).

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

For a significant performance improvement on Intel-based computers,
I would recommend the official [IntelPython distribution][1]. This
distribution includes a SciPy variant that uses the Intel Math Kernel
Library as its backend, which significantly speeds up e.g. eigenvalue
computations. To use this, follow Intel's official instructions and
then install the dependencies listed in `pyproject.toml`. Afterwards,
simply run the various Bodge scripts directly in that environment.

Another recommendation would be to install [CuPy][2], which is an
optional dependency of this project. On machines with nVidia GPUs,
you can then provide a `cuda=True` flag to some computational
methods in order to perform GPU instead of CPU computations.

[1]: https://www.intel.com/content/www/us/en/developer/tools/oneapi/distribution-for-python.html#gs.h2ajdj
[2]: https://cupy.dev/



