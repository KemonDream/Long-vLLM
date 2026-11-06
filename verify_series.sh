#!/usr/bin/env bash
# Verify that the published patch series reproduces the published file set
# exactly. Run this before trusting a release: it is the check that matters.
#
# Usage:  bash verify_series.sh /path/to/vllm-checkout [base-commit]
#
# It creates a temporary git worktree at the pinned base commit, applies every
# patch in order (exit code enforced), then byte-compares every file in
# MANIFEST.tsv against the applied result. Requires a git checkout of vLLM that
# contains the base commit.
set -uo pipefail

SRC="${1:?usage: verify_series.sh /path/to/vllm-checkout [base-commit]}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Default to the documented upstream v0.31.0 commit: a reader applies the series
# to upstream's tree, so that is what reproduction must target. Pass a second
# argument only to reproduce against a local snapshot instead.
UPSTREAM=db9527a46873454610df6dbedf79a36d6bf1a7f6
BASE="${2:-$UPSTREAM}"
WT="$(mktemp -d /tmp/sm70verify.XXXXXX)"

cd "$SRC" || exit 1
[ "$BASE" = "$UPSTREAM" ] || echo "note: base $BASE is not upstream v0.31.0 ($UPSTREAM)"
git worktree add -f -q "$WT" "$BASE" || { echo "FAIL: cannot create worktree at $BASE"; exit 1; }
echo "worktree at $WT ($(cd "$WT" && git rev-parse --short HEAD))"

fail=0
for p in "$HERE"/patches/*.patch; do
  if git -C "$WT" apply --index --3way "$p" >/tmp/_verify_apply.log 2>&1; then
    printf 'apply  OK   %s\n' "$(basename "$p")"
  else
    printf 'apply  FAIL %s\n' "$(basename "$p")"; head -3 /tmp/_verify_apply.log; fail=1
  fi
done

missing=0; differing=0; checked=0; deleted=0; undeleted=0
while IFS=$'\t' read -r st path prov; do
  [ "$path" = "path" ] && continue
  [ -z "$path" ] && continue
  if [ "$st" = "D" ]; then
    # A deletion must actually happen, or "the series reproduces the set" is
    # satisfied by patches that quietly leave a file in place.
    deleted=$((deleted+1))
    [ -e "$WT/$path" ] && { echo "NOT DELETED BY PATCHES: $path"; undeleted=$((undeleted+1)); }
    continue
  fi
  checked=$((checked+1))
  if [ ! -f "$HERE/files/$path" ]; then echo "MISSING FROM SET: $path"; missing=$((missing+1)); continue; fi
  if [ ! -f "$WT/$path" ]; then echo "NOT PRODUCED BY PATCHES: $path"; missing=$((missing+1)); continue; fi
  cmp -s "$HERE/files/$path" "$WT/$path" || { echo "DIFFERS: $path"; differing=$((differing+1)); }
done < "$HERE/MANIFEST.tsv"

echo "checked=$checked  not-produced=$missing  differing=$differing  deletions=$deleted (undeleted=$undeleted)"
git -C "$SRC" worktree remove --force "$WT" 2>/dev/null
[ $fail -eq 0 ] && [ "$missing" -eq 0 ] && [ "$differing" -eq 0 ] && [ "$undeleted" -eq 0 ] \
  && { echo "VERDICT: the series reproduces the published set exactly"; exit 0; } \
  || { echo "VERDICT: FAILED - do not publish"; exit 1; }
