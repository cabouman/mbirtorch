Package Maintenance
===================

The following describes procedures for basic package maintenance.

Unit Tests
----------

In the ``mbirtorch`` conda environment::

    dev_scripts/run_tests.sh

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

2. Check the TestPyPI upload.  Make a clean conda environment with the
   release candidate and run the tests::

       dev_scripts/make_test_environment.sh 0.X.Yrc1
       conda activate test
       dev_scripts/run_tests.sh

   If a test fails, fix the problem and repeat from step 1 with ``0.X.Yrc2``.

Releasing to PyPI
+++++++++++++++++

This procedure stands on its own.  The TestPyPI steps above are optional.

1. Open the release pull request::

       dev_scripts/release.sh 0.X.Y

   What this does:

   * Sets ``__version__`` to 0.X.Y, commits, and pushes to ``prerelease``.
   * Opens the pull request from ``prerelease`` to ``main``.
   * Nothing is uploaded anywhere.

   Next: review the pull request on GitHub.  The tests run on it automatically,
   so their result shows next to the Merge button.  When you are happy, accept
   the pull request from ``prerelease`` to ``main``.

2. Publish the release::

       dev_scripts/release.sh 0.X.Y --publish

   What this does:

   * Checks that ``main`` contains ``__version__ = 0.X.Y``; stops if the
     pull request is not merged yet.
   * Fast-forwards ``prerelease`` up to ``main`` so the two branches are
     identical, and updates your local ``main`` to match.
   * Creates a GitHub release with tag ``v0.X.Y`` on the shared commit.
   * CI builds the package and publishes it to PyPI.  No approval step.

   After this, ``main``, ``prerelease``, and the ``v0.X.Y`` tag are all on the
   same commit.

3. Check the PyPI upload.  Make a clean conda environment with the new
   version and run the tests::

       dev_scripts/make_test_environment.sh
       conda activate test
       dev_scripts/run_tests.sh

