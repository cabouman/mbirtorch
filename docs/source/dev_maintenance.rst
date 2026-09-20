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

   Next: Once the checks pass on GitHub, accept the pull request from prerelease to main.

2. Publish the release::

       dev_scripts/release.sh 0.X.Y --publish

   What this does:

   * Checks that ``main`` contains ``__version__ = 0.X.Y``; stops if the
     pull request is not merged yet.
   * Creates a GitHub release with tag ``v0.X.Y`` on ``main``.
   * CI builds the package, then pauses and waits for your approval.

   Next: You must next approve the deployment on GitHub.
   To do this: On GitHub, open the Actions tab, click the running release
   workflow, click "Review deployments", check the "pypi" box, and click "Approve and deploy".

3. Check the PyPI upload.  Make a clean conda environment with the new
   version and run the tests::

       dev_scripts/make_test_environment.sh
       conda activate test
       dev_scripts/run_tests.sh

