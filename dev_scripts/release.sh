#!/bin/bash
# Run one stage of the release procedure in dev_maintenance.rst.
#
#   dev_scripts/release.sh 0.2.0rc1           # rc: publish a pre-release to TestPyPI
#   dev_scripts/release.sh 0.2.0              # final: open the pull request to main
#   dev_scripts/release.sh 0.2.0 --publish    # after main advances: publish to PyPI
#
# Requires the gh CLI, logged in.  Uploads still need approval of the pypi
# environment on the workflow run page.
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
  # The tag must point at main, so main must already carry this version.
  git fetch -q origin main
  if ! git show origin/main:$INIT | grep -q "__version__ = \"$VERSION\""; then
    echo "main does not have __version__ = \"$VERSION\"; merge the PR first" >&2
    exit 1
  fi
  gh release create "v$VERSION" --target main --title "MBIRTorch v$VERSION" \
    --generate-notes
  echo "Release v$VERSION created.  Approve the pypi environment on the"
  echo "workflow run page, then check with:"
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
