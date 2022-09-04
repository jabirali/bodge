# Bodge

Bodge is a Python solver for large-scale tight-binding models. Although quite
general models of this type are supported, we target the BOgoliubov-DeGEnnes
equations for superconductivity, which is where the code name comes from.

## Development

To develop this code, it is recommended that you install `pytest`, `isort`,
`black`, and `taskfile`. You can then run `task -w` in a terminal to ensure
that any code changes are consistent with the style of the existing code, and
check that previous assumptions about the code's behavior are not violated.