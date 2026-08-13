Package Maintenance
===================

The following describes procedures for basic package maintenance.

Unit Tests
----------

From the repository root, in the ``mbirtorch`` conda environment::

    python -m pytest -n 4 tests ci

To include the cross-framework parity tests against mbirjax, first generate
the golden archives (``tests/generate_goldens.py``, run in the mbirjax
environment), then::

    python -m pytest -m "goldens or not goldens" tests

The same tests run automatically on every push and pull request.

Releasing a New Version
-----------------------

This is only available for registered maintainers.  It requires the ``gh``
command, logged in to GitHub.  The example below releases version 0.X.Y.

Releasing to TestPyPI
+++++++++++++++++++++

1. Publish a release candidate to TestPyPI::

       dev_scripts/release.sh 0.X.Yrc1

2. Check the TestPyPI upload::

       dev_scripts/check_published_wheel.sh --testpypi --version 0.X.Yrc1

   If it fails, fix the problem and repeat from step 1 with ``0.X.Yrc2``.

Releasing to PyPI
+++++++++++++++++

3. Open the release pull request::

       dev_scripts/release.sh 0.X.Y

   Merge it on GitHub when the checks pass.

4. Publish the release::

       dev_scripts/release.sh 0.X.Y --publish

   Then approve the ``pypi`` environment on the workflow run page.  This
   uploads to PyPI.

5. Check the PyPI upload::

       dev_scripts/check_published_wheel.sh --version 0.X.Y

Each ``release.sh`` stage sets ``__version__`` in ``mbirtorch/__init__.py``,
commits, and creates the matching ``v``-prefixed tag; the upload fails if the
tag and ``__version__`` ever disagree.  The manual procedure behind the
script: edit ``__version__``, commit to ``prerelease``, and draft a GitHub
release with tag ``v`` + ``__version__`` — target ``prerelease`` with "Set as
a pre-release" checked for an rc, target ``main`` for a final version.

The documentation rebuilds automatically: ``latest`` follows ``main``, and
``stable`` follows the highest release tag.

Notes
-----

* Uploads use PyPI Trusted Publishing; no token or password is stored.
* The tested Python versions are in ``.github/python-versions.json``.  A
  nightly check opens a pull request when torch's supported versions change;
  merging it is the whole update.
* Manual upload with ``twine`` remains available as a fallback.
