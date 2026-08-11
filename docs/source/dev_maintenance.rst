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

1. Update ``__version__`` in ``mbirtorch/__init__.py`` and merge to
   ``prerelease``.  This is the only place the version number is written.

2. On GitHub, draft a new release: tag ``vX.Y.ZrcN``, target ``prerelease``,
   check "Set as a pre-release", and publish.  This uploads to TestPyPI.

3. Check the TestPyPI upload::

       dev_scripts/check_published_wheel.sh --testpypi --version X.Y.ZrcN

4. Open a pull request from ``prerelease`` to ``main`` and merge it when the
   checks pass.

5. Draft a new release: tag ``vX.Y.Z``, target ``main``, and publish.  Then
   approve the ``pypi`` environment on the workflow run page.  This uploads
   to PyPI.

6. Check the PyPI upload::

       dev_scripts/check_published_wheel.sh --version X.Y.Z

The documentation rebuilds automatically: ``latest`` follows ``main``, and
``stable`` follows the highest release tag.

Notes
-----

* Uploads use PyPI Trusted Publishing; no token or password is stored.
* The tested Python versions are in ``.github/python-versions.json``.  A
  nightly check opens a pull request when torch's supported versions change;
  merging it is the whole update.
* Manual upload with ``twine`` remains available as a fallback.
