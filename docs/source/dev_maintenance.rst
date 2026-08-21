Package Maintenance
===================

The following describes procedures for basic package maintenance.

Unit Tests
----------

From the repository root, in the ``mbirtorch`` conda environment::

    python -m pytest -n 4 tests ci

To include the cross-framework parity tests against mbirjax, first generate
the golden archives.  Two scripts write them, both run in the mbirjax
environment::

    python tests/generate_goldens.py
    python tests/generate_preprocess_goldens.py

Then run the full suite::

    python -m pytest -m "goldens or not goldens" tests

A parity test whose archive is missing skips rather than fails, so run both
scripts to get the whole set.

The same tests run automatically on every push and pull request.

Releasing a New Version
-----------------------

This is only available for registered maintainers.  It requires the ``gh``
command, logged in to GitHub.  The example below releases version 0.X.Y.

Releasing to TestPyPI
+++++++++++++++++++++

1. Publish a release candidate to TestPyPI::

       dev_scripts/release.sh 0.X.Yrc1

   What this does:

   * Sets ``__version__`` to 0.X.Yrc1, commits, and pushes to ``prerelease``.
   * Creates a GitHub pre-release with tag ``v0.X.Yrc1``.
   * CI builds the package and uploads it to TestPyPI.  No approval needed.

2. Check the TestPyPI upload::

       dev_scripts/check_published_wheel.sh --testpypi --version 0.X.Yrc1

   If it fails, fix the problem and repeat from step 1 with ``0.X.Yrc2``.

Releasing to PyPI
+++++++++++++++++

3. Open the release pull request::

       dev_scripts/release.sh 0.X.Y

   What this does:

   * Sets ``__version__`` to 0.X.Y, commits, and pushes to ``prerelease``.
   * Opens the pull request from ``prerelease`` to ``main``.
   * Nothing is uploaded anywhere.

   Merge the pull request on GitHub when the checks pass.

4. Publish the release::

       dev_scripts/release.sh 0.X.Y --publish

   What this does:

   * Checks that ``main`` contains ``__version__ = 0.X.Y``; stops if the
     pull request is not merged yet.
   * Creates a GitHub release with tag ``v0.X.Y`` on ``main``.
   * CI builds the package, then pauses and waits for your approval.

   To approve: on GitHub, open the Actions tab, click the running release
   workflow, click "Review deployments", check the "pypi" box, and click
   "Approve and deploy".  The upload to PyPI then runs.  This manual
   approval is the last stop before PyPI, where uploads are permanent.

5. Check the PyPI upload::

       dev_scripts/check_published_wheel.sh --version 0.X.Y

The three ``release.sh`` commands divide the work.  Steps 1 and 3 set
``__version__`` in ``mbirtorch/__init__.py``, commit it, and push to
``prerelease``.  Steps 1 and 4 create the matching ``v``-prefixed tag, on
``prerelease`` and on ``main`` respectively.  Step 4 changes no file: it checks
that ``main`` already carries the version and stops if it does not.  The upload
fails if the tag and ``__version__`` ever disagree.

The script automates a short manual procedure: edit ``__version__``, commit to
``prerelease``, and draft a GitHub release with tag ``v`` + ``__version__``.
For an rc, target ``prerelease`` with "Set as a pre-release" checked.  For a
final version, target ``main``.

The documentation rebuilds automatically: ``latest`` follows ``main``, and
``stable`` follows the highest release tag.

Notes
-----

* Uploads use PyPI Trusted Publishing; no token or password is stored.
* The tested Python versions are in ``.github/python-versions.json``.  A
  nightly check opens a pull request when torch's supported versions change;
  merging it is the whole update.
* Manual upload with ``twine`` remains available as a fallback.
