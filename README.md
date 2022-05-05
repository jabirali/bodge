# Bodge

Bodge is a linear-scaling solver for megadimensional tight-binding models.
Although quite general tight-binding models are supported, we especially
target the BOgoliubov-DeGEnnes equation for ballistic superconductivity,
which is where the name comes from. The solver is implemented in Python,
and is based on the Local Chebyshev expansion of the Green function.

## Development

To develop this code, it is recommended that you install `pytest`, `isort`,
`black`, and `taskfile`. You can then run `task -w` in a terminal to ensure
that any code changes are consistent with the style of the existing code, and
check that previous assumptions about the code's behavior are not violated.