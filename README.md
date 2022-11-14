# Bodge
Bodge is a Python solver for large-scale tight-binding models. Although quite
general models of this type are supported, we target the BOgoliubov-DeGEnnes
equations for superconductivity, which is where the project name comes from.

## Installation
To use the default Python version on your system, simply run:

	make install

If you prefer to use a newer Python version (recommended), first install
this via your package manager of choice. For example:

	brew install python@3.11          # macOS with HomeBrew
	sudo apt install python3.11-full  # Ubuntu GNU/Linux

Afterwards, mention what Python version to use when installing Bodge:

	make install PYTHON=python3.11

Run `make` without any command-line arguments to see how to proceed further.
This should provide information on how to run the bundled unit tests, run
scripts that use the Bodge package, or run the autoformatter after updates.

For a significant performance improvement on Intel-based computers,
I would recommend the official [IntelPython distribution][1]. This
distribution includes a SciPy variant that uses the Intel Math Kernel
Library as its backend, which significantly speeds up e.g. eigenvalue
computations. To use this, follow Intel's official instructions and
then install the dependencies listed in `pyproject.toml`. Afterwards,
simply run the various Bodge scripts directly in that environment.

[1]: https://www.intel.com/content/www/us/en/developer/tools/oneapi/distribution-for-python.html#gs.h2ajdj
