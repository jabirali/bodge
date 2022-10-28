all: help
.PHONY: help install test format clean .FORCE

define help
Usage:
	make <target> [flags]

Targets:
	help        Show this help message
	install     Install the package into a virtual environment
	test        Execute the unit tests bundled with the project
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
endef
export help

PYTHON = python3

help:
	@echo "$$help"

install:
	test -d venv || $(PYTHON) -m venv venv
	. venv/bin/activate; pip install --upgrade pip; pip install --prefer-binary numpy; pip install --prefer-binary -e .

test:
	. venv/bin/activate; python -m pytest tests

format:
	. venv/bin/activate; python -m isort .
	. venv/bin/activate; python -m black .

clean:
	rm -rf venv
	find . -iname "*.pyc" -delete
	find . -iname "__pycache__" -delete

%.py: .FORCE
	. venv/bin/activate; python $@