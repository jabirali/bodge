# Bodge
Bodge is a Python solver for large-scale tight-binding models. Although quite
general models of this type are supported, we target the BOgoliubov-DeGEnnes
equations for superconductivity, which is where the code name comes from.

## Installation (General)
To use the default Python version on your system, simply run:

	make install

If you prefer to use a newer Python version (recommended), first install
this via your package manager of choice. For example:

	brew install python@3.11          # macOS with HomeBrew
	sudo apt install python3.11-full  # Ubuntu GNU/Linux

Afterwards, mention what Python version to use when installing Bodge:

	make install PYTHON=python3.11

Run `make` without command-line arguments to see how to proceed further.

## Installation (IntelPython)
For the best performance on Intel-based computers, it is recommended to use
the official [IntelPython distribution][1]. These use the Intel Math Kernel
Library as its backend, which provides much faster linear algebra routines.
In that case, follow the official instructions, and then use `conda install`
to manually add dependencies listed in `pyproject.toml` to your environment.
Afterwards, simply run the various Bodge scripts directly in that environment.

[1]: https://www.intel.com/content/www/us/en/developer/tools/oneapi/distribution-for-python.html#gs.h2ajdj