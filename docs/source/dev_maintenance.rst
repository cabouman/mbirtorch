Package Maintenance
===================

The following describes procedures for basic package maintenance.

Unit Tests
----------

In the ``mbirtorch`` conda environment::

    dev_scripts/run_tests.sh

Releasing a version
-------------------

mbirtorch is published to PyPI by the GitHub Actions workflow in
``.github/workflows/release.yml``.  You drive it from your machine with
``dev_scripts/release.sh``: one command opens a pull request for you to review,
and after you merge it a second command tags the release and publishes it.
Uploads use Trusted Publishing, so no API token is ever stored or typed.

The examples release version ``0.X.Y``.  Replace ``0.X.Y`` with the version you
are releasing.

Dry run on TestPyPI (optional, recommended the first time)
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

1. Publish a release candidate::

       dev_scripts/release.sh 0.X.Yrc1

   Stamps the version, pushes ``prerelease``, and creates a pre-release tagged
   ``v0.X.Yrc1``.  CI builds the package and uploads it to TestPyPI.  No approval
   is needed.

2. Check the upload::

       pip install -i https://test.pypi.org/simple/ mbirtorch

Release to PyPI
+++++++++++++++

1. Open the release pull request::

       dev_scripts/release.sh 0.X.Y

   Sets the version on ``prerelease``, pushes it, and opens the pull request to
   ``main``.  Nothing is published yet.

2. Review the pull request on GitHub.  The tests run on it automatically, so
   their result shows next to the Merge button.  When you are happy, merge it.

3. Publish the release::

       dev_scripts/release.sh 0.X.Y --publish

   Fast-forwards ``prerelease`` up to ``main``, updates your local ``main`` to
   match, tags the shared commit ``v0.X.Y``, and CI builds and publishes it to
   PyPI.  No approval step.

4. Confirm it is live::

       pip install mbirtorch

Notes
+++++

- After step 3, ``main``, ``prerelease``, and the ``v0.X.Y`` tag are all on the
  same commit.
- The tag is always ``v`` followed by the version; the build fails if the tag
  does not match ``__version__``.
- The version is single-sourced from the package's ``__init__.py``.
