# Makefile metadata.
all: help
.PHONY: help install test shell format clean .FORCE
.PRECIOUS: %.py

# Default values for flags.
PYTHON = python3

# Define the interactive help message.
define help
Usage:
	make <target> [flags]

Targets:
	help        Show this help message
	install     Install the package into a virtual environment
	test        Execute the unit tests bundled with the project
	shell       Open an iPython shell in the virtual environment
	format      Reformat source code to fit the project style
	clean       Remove virtual environment and temporary files
	<filename>  Run the given Python script in the environment

Flags:
	PYTHON      Python version to use for the `install` process
	            (defaults to the `python3` in your system path)

Examples:
	make install PYTHON=python3.11   # Install with Python 3.11
	make test                        # Test that everything works
	make pwave.py                    # Run `pwave.py` using Bodge

Note that `make install` only installs the project dependencies;
additional development dependencies are installed as needed.
endef
export help

help:
	@echo "$$help"

# Basic user interface for the makefile.
install:
	test -d venv || $(PYTHON) -m venv venv
	. venv/bin/activate; pip install --upgrade pip; pip install --prefer-binary numpy; pip install --prefer-binary -e .

test: venv/bin/pytest
	. venv/bin/activate; pytest tests

shell: venv/bin/ipython
	. venv/bin/activate; ipython

format: venv/bin/isort venv/bin/black
	. venv/bin/activate; isort .; black .

clean:
	rm -rf venv
	find . -iname "*.pyc" -delete
	find . -iname "__pycache__" -delete

# Automatically install development dependencies as needed.
venv/bin/pytest:
	. venv/bin/activate; pip install pytest

venv/bin/black:
	. venv/bin/activate; pip install black

venv/bin/isort:
	. venv/bin/activate; pip install isort

venv/bin/ipython:
	. venv/bin/activate; pip install ipython

# Run Python scripts inside the virtual environment.
%.py: .FORCE
	. venv/bin/activate; python $@