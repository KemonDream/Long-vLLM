#!/usr/bin/env bash
# Apply this patch set onto a clean upstream vLLM checkout.
#
# Usage:  bash apply.sh /path/to/clean/vllm-v0.31.0
#
# The series is generated against upstream tag v0.31.0
# (db9527a46873454610df6dbedf79a36d6bf1a7f6). It will NOT apply cleanly to any
# other base - check out that tag first.
set -euo pipefail

TREE="${1:?usage: apply.sh /path/to/clean/vllm-v0.31.0}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE=db9527a46873454610df6dbedf79a36d6bf1a7f6

# `-d .git` is wrong for two common checkouts: a git worktree and a submodule
# both have .git as a FILE. Ask git instead. (Found by running this on a
# worktree, which is exactly how it is used here.)
if ! git -C "$TREE" rev-parse --git-dir >/dev/null 2>&1; then
  echo "error: $TREE is not a git checkout" >&2
  exit 1
fi

cd "$TREE"
head="$(git rev-parse HEAD)"
if [ "$head" != "$BASE" ] && ! git merge-base --is-ancestor "$BASE" HEAD 2>/dev/null; then
  echo "warning: HEAD ($head) is not upstream v0.31.0 ($BASE)." >&2
  echo "         patches are generated against that commit; expect conflicts." >&2
fi

for p in "$HERE"/patches/*.patch; do
  echo "== applying $(basename "$p")"
  git apply --index --3way "$p"
done

# Release documents are additive: they live in the repository root of the
# published set rather than inside the vLLM tree.
# LICENSE and CHANGELOG.md belong in this list: NOTICE says "See LICENSE for the
# full text", so copying NOTICE without it leaves a dangling pointer in the tree.
for f in README.md KNOWN-LIMITS.md NOTICE PROVENANCE.md LICENSE CHANGELOG.md; do
  [ -f "$HERE/$f" ] && cp "$HERE/$f" "$TREE/$f"
done

cat <<'EOF'

Applied. Next:
  1. bash build_sm70.sh <this-tree>
  2. read KNOWN-LIMITS.md before selecting a KV cache dtype
  3. do not select turboquant_k8v4 on Volta
EOF
