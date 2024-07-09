# Makefile metadata.
all: help
.PHONY: help install test format clean .FORCE
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
	docs        Regenerate the documentation using Quarto
	test        Execute the unit tests bundled with the project
	format      Reformat source code to fit the project style
	clean       Remove virtual environment and temporary files
	<filename>  Run a Python script in the virtual environment

Flags:
	PYTHON      Python version to use for the `install` process
	            (defaults to the `python3` in your system path)

Examples:
	make install PYTHON=python3.11   # Install with Python 3.11
	make test                        # Test that everything works
	make script.py                   # Run `script.py` using Bodge
endef
export help

help:
	@echo "$$help"

# Basic user interface for the makefile.
install:
	test -d venv || $(PYTHON) -m venv venv
	. venv/bin/activate; pip install --upgrade pip; pip install --prefer-binary numpy; pip install --prefer-binary --editable .[dev]

docs: .FORCE
	. venv/bin/activate; quarto render

test:
	. venv/bin/activate; pytest tests

format:
	. venv/bin/activate; isort .; black .

shell:
	. venv/bin/activate; exec $(SHELL)

clean:
	rm -rf venv
	find . -iname "*.pyc" -delete
	find . -iname "__pycache__" -delete


# Run Python scripts inside the virtual environment.
%.py: .FORCE
	. venv/bin/activate; python $@
