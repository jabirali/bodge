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
- [ ] Check which functions can be CUDA-enhanced without too much hassle
- [X] Refactor *wave functions into Hamiltonian
- [ ] Consider whether to keep the SSD method. If we do, it might be cleaner to give an `ssd=True` flag to the Hamiltonian matrix itself to enable it during Hamiltonian population, and use a method `_ssd` to implement it. That way, `H[i, j]` etc. would automatically implement this stuff for us.
## [ ] Tests
- [ ] Get close to 100% test coverage
- [ ] Read through existing tests to check that they're reasonable
# Scripts
- [ ] Determine which scripts to include as docs. Purge them from the root folder.
# Documentation
Modern development practices:
- [ ] Well-documented – extensive use of docstrings, comments, and a tutorial.
- [ ] Type hints for core libraries
- [ ] Docstrings for all classes/functions in core libraries
- [ ] Beartype for runtime type checking of all core libraries
- [ ] Pytest for unit testing of all core functionality
- [ ] GitHub actions for continuous integration (ensuring that it works OK across Python 3.9-3.12)

Todo:
- [ ] Note about using Greek letters in examples. (Allowed by Python, keeps code closer to math, but everything can be used without this.)
- [ ] Usage examples for swave / pwave / dwave
# Scaffolding
- environment.yml and/or pip package?
- Consider getting rid of unneeded dependencies (but it's probably fine)