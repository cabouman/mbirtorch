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

2. Check the TestPyPI upload.  Make a clean conda environment, install the
   release candidate into it from TestPyPI, and run the tests::

       source dev_scripts/make_test_environment.sh
       pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple "mbirtorch[test]==0.X.Yrc1"
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

3. Check the PyPI upload.  Make a clean conda environment, install the
   package into it from PyPI, and run the tests::

       source dev_scripts/make_test_environment.sh
       dev_scripts/run_tests.sh

   The first script creates a conda environment named ``test`` and installs
   ``mbirtorch[test]`` from PyPI.  Confirm that it picked up the new version::

       python -c "import mbirtorch; print(mbirtorch.__version__)"

