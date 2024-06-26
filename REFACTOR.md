# Module
## [X] common.py
- [X] Move most imports to =utils.py=
## [X] hamiltonian.py
- [X] Clean the class and docstrings
## [X] lattice.py
- [X] Clean the Lattice class
- [X] Clean the CubicLattice class
## [ ] utils.py
- [ ] Check: renaming, refactoring, testing, docstrings, and return formats (for ldos)
- [X] Check which functions can be CUDA-enhanced without too much hassle
- [X] Refactor *wave functions into Hamiltonian
- [X] Refactor the ssd method into Hamiltonian
## [ ] Tests
- [ ] Get close to 100% test coverage
- [ ] Read through existing tests to check that they're reasonable
- [ ] Double-cheeck that all the unit tests are in the right place
# Scripts
- [ ] Determine which scripts to include as docs. Purge them from the root folder.
# Documentation
Modern development practices:
- [ ] Well-documented – extensive use of docstrings, comments, and a tutorial.
- [ ] Type hints for core libraries
- [ ] Docstrings for all classes/functions in core libraries
- [ ] Beartype for runtime type checking of all core libraries
- [ ] Pytest for unit testing of all core functionality
- [ ] GitHub actions for continuous integration (ensuring that it works OK across Python 3.10-3.12)
    - [ ] Skip the `environment.yml` and focus on the PyPi package.
    - [ ] Keep the `Makefile`, it's needed for GitHub Actions.
    - [ ] Look if we can then use `match` syntax again.

Todo:
- [ ] Note about using Greek letters in examples. (Allowed by Python, keeps code closer to math, but everything can be used without this.)
- [ ] Usage examples for swave / pwave / dwave
# Scaffolding
- environment.yml and/or pip package?
- Consider getting rid of unneeded dependencies (but it's probably fine)
