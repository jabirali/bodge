# Contributing to Bodge

I'm happy to receive contributions via GitHub pull requests. For less
"complete" contributions, feel free to either open a GitHub issue or
contact me via [email](mailto:jabir.ali.ouassou@hvl.no).

## Installation

While `pip install bodge` is sufficient as a "Bodge user", for
development I would recommend that you clone the Git repository
directly and install it into a separate virtual environment.

The easiest way to do so from a Unix-like terminal is as follows:
```sh
git clone git@github.com:jabirali/bodge.git
cd bodge
make install
```

This will create a virtual environment in a subfolder called `venv`,
and then install Bodge into that virtual environment. Once this is in
place, you can start working on the Bodge source code.

## Unit testing

Bodge comes with a set of unit tests that check if everything
works. This is done via the `pytest` framework, and the tests
themselves can be found in the `tests` subfolder. To run these:

	make test

I'd recommend running these tests after installation (to check that
your dev environment is correctly setup), and then again before each
Git commit you perform (to ensure nothing is broken in the committed
version of the code). Naturally, some changes to the code do require
that the tests be modified, in which case I'd appreciate suggested
changes to the unit tests along with the contributed new code.

## Running scripts

To test the new code you're implementing, you likely want to run some
example scripts (that do `from bodge import *` and then perform some
calculations) separately from the unit tests discussed above.

If you call such a script e.g. `example.py`, you can run it inside the
virtual environment created above using the `Makefile` as follows:

	make example.py

## Formatting source code

This project uses the autoformatters `black` and `isort` to ensure
that the Python code follows a consistent and readable style.
Following that style is easy: Just run the following command before
each time you perform a Git commit (at least before a pull request):

	make format

Alternatively, most Python editors and IDEs have plugins that can run
`black` and `isort` automatically each time you save the file.

## Documentation

Please include a short "docstring" in every new function that is
implemented, which describes briefly what that function does.

If you wish, you can consider also contributing an example of how to
use your contribution in the official documentation. To do so, modify
the file `tutorial.qmd` and run `make docs` afterwards (can be slow).

## Submitting upstream

Once you have a change you want to share with others, feel free to
submit a "pull request" on GitHub and I'd be happy to have a look.

Please include some context: why/when this change is useful, and
ideally also an example that shows its intended usage.
