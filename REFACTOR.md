# Module
## [ ] common.py
- [X] Move most imports to =utils.py=
- [ ] Consider getting rid of unneeded dependencies (but it's probably fine)
## [X] hamiltonian.py
- [X] Clean the class and docstrings
## [ ] lattice.py
- [X] Clean the Lattice class
- [X] Clean the CubicLattice class
## [ ] utils,py
- [ ] Consider renaming and refactoring
# Scripts
- [ ] Determine which scripts to include
# Documentation
Modern development practices:
- [ ] Well-documented – extensive use of docstrings, comments, and a tutorial.
- [ ] Type hints for core libraries
- [ ] Docstrings for all classes/functions in core libraries
- [ ] Beartype for runtime type checking of all core libraries
- [ ] Pytest for unit testing of all core functionality
- [ ] GitHub actions for continuous integration (ensuring that it works across Python 3.9-3.11)
# Scaffolding
- environment.yml and/or pip package?