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

This is only available for registered maintainers.

The tag must always equal ``v`` + ``__version__``; the upload fails otherwise.
The example below releases version 0.2.0.

1. In ``mbirtorch/__init__.py``, set ``__version__ = "0.2.0rc1"``.
   Commit this to the ``prerelease`` branch.

2. On GitHub, draft a new release: tag ``v0.2.0rc1``, target ``prerelease``.
   Check "Set as a pre-release", and publish.  This uploads to TestPyPI.

3. Check the TestPyPI upload::

       dev_scripts/check_published_wheel.sh --testpypi --version 0.2.0rc1

   If it fails, fix the problem, set ``__version__ = "0.2.0rc2"``, and repeat
   from step 2.

4. Set ``__version__ = "0.2.0"`` and commit to ``prerelease``.  Open a pull
   request from ``prerelease`` to ``main`` and merge it when the checks pass.

5. Draft a new release: tag ``v0.2.0``, target ``main``, and publish.  Then
   approve the ``pypi`` environment on the workflow run page.  This uploads
   to PyPI.

6. Check the PyPI upload::

       dev_scripts/check_published_wheel.sh --version 0.2.0

The documentation rebuilds automatically: ``latest`` follows ``main``, and
``stable`` follows the highest release tag.

Notes
-----

* Uploads use PyPI Trusted Publishing; no token or password is stored.
* The tested Python versions are in ``.github/python-versions.json``.  A
  nightly check opens a pull request when torch's supported versions change;
  merging it is the whole update.
* Manual upload with ``twine`` remains available as a fallback.
