#!/bin/bash
# Run one stage of the release procedure in dev_maintenance.rst.
#
#   dev_scripts/release.sh 0.2.0rc1           # rc: publish a pre-release to TestPyPI
#   dev_scripts/release.sh 0.2.0              # final: open the pull request to main
#   dev_scripts/release.sh 0.2.0 --publish    # after main advances: publish to PyPI
#
# Requires the gh CLI, logged in.  GitHub Actions publishes to PyPI with no
# manual approval step.
set -euo pipefail

cd "$(dirname "$0")/.."
VERSION="${1:?usage: release.sh X.Y.Z[rcN] [--publish]}"
PUBLISH="${2:-}"
INIT=mbirtorch/__init__.py

case "$VERSION" in
  *rc*) STAGE=rc ;;
  *)    STAGE=final ;;
esac
if [[ "$PUBLISH" == "--publish" && "$STAGE" == "rc" ]]; then
  echo "--publish is for a final version; an rc publishes on its own" >&2
  exit 2
fi

if [[ "$PUBLISH" == "--publish" ]]; then
  # The PR is merged, so main carries this version.  Bring prerelease up to
  # main, then tag the shared commit and let CI publish.
  git fetch -q origin main
  if ! git show origin/main:$INIT | grep -q "__version__ = \"$VERSION\""; then
    echo "main does not have __version__ = \"$VERSION\"; merge the PR first" >&2
    exit 1
  fi
  # Fast-forward prerelease to main so the two branches are identical.
  git checkout -q prerelease
  git pull -q origin prerelease
  if ! git merge -q --ff-only origin/main; then
    echo "could not fast-forward prerelease to main; merge the PR with the" >&2
    echo "default 'Create a merge commit' option, then run --publish again" >&2
    exit 1
  fi
  git push -q origin prerelease
  git branch -f main origin/main        # local main to the same commit
  gh release create "v$VERSION" --target main --title "MBIRTorch v$VERSION" \
    --generate-notes
  echo "Release v$VERSION created.  main, prerelease, and the v$VERSION tag are"
  echo "all on the same commit.  GitHub Actions is publishing to PyPI; check with:"
  echo "  dev_scripts/check_published_wheel.sh --version $VERSION"
  exit 0
fi

git checkout -q prerelease
git pull -q origin prerelease
sed -i '' "s/^__version__ = \".*\"/__version__ = \"$VERSION\"/" $INIT
grep -q "__version__ = \"$VERSION\"" $INIT
# Stamp the version and date into CITATION.cff and the BibTeX entries.
python3 dev_scripts/update_citation.py
git add $INIT CITATION.cff README.md docs/source/credits.rst docs/source/refs.bib
git commit -q -m "Set version to $VERSION"
git push -q origin prerelease

if [[ "$STAGE" == "rc" ]]; then
  gh release create "v$VERSION" --target prerelease --prerelease \
    --title "MBIRTorch v$VERSION" --generate-notes
  echo "Pre-release v$VERSION created; TestPyPI upload is running.  Check with:"
  echo "  dev_scripts/check_published_wheel.sh --testpypi --version $VERSION"
else
  # main changes only through a pull request, merged on GitHub.
  if gh pr list --base main --head prerelease --state open --json number -q '.[0].number' | grep -q .; then
    echo "The pull request from prerelease to main is already open and now carries $VERSION."
  else
    gh pr create --base main --head prerelease --title "MBIRTorch v$VERSION" \
      --body "Release v$VERSION."
  fi
  echo "When the checks pass, merge the pull request on GitHub.  Then run:"
  echo "  dev_scripts/release.sh $VERSION --publish"
fi
