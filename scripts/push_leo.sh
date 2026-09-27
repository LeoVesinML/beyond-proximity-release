#!/usr/bin/env bash
# Push only lightweight files (README, docs/assets) — fast, no large binaries.
set -euo pipefail
cd "$(dirname "$0")/.."

REMOTE="${1:-https://github.com/LeoPython2006/beyond-proximity.git}"
BRANCH="${2:-feature/auto-query-pipeline}"

echo "→ Pushing branch ${BRANCH} to LeoPython2006/beyond-proximity ..."
git push "$REMOTE" "HEAD:${BRANCH}"
echo "✓ Done. Open: https://github.com/LeoPython2006/beyond-proximity"
