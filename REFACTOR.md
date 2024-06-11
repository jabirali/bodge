# Module
## [X] common.py
- [X] Move most imports to =utils.py=
## [X] hamiltonian.py
- [X] Clean the class and docstrings
## [X] lattice.py
- [X] Clean the Lattice class
- [X] Clean the CubicLattice class
## [ ] utils,py
- [ ] Consider renaming and refactoring
# Scripts
- [ ] Determine which scripts to include as docs. Purge them from the root folder.
# Documentation
Modern development practices:
- [ ] Well-documented – extensive use of docstrings, comments, and a tutorial.
- [ ] Type hints for core libraries
- [ ] Docstrings for all classes/functions in core libraries
- [ ] Beartype for runtime type checking of all core libraries
- [ ] Pytest for unit testing of all core functionality
- [ ] GitHub actions for continuous integration (ensuring that it works across Python 3.9-3.11)

Todo:
- [ ] Note about using Greek letters in examples. (Allowed by Python, keeps code closer to math, but everything can be used without this.)
# Scaffolding
- environment.yml and/or pip package?
- Consider getting rid of unneeded dependencies (but it's probably fine)