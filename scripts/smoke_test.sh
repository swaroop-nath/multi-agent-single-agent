#!/usr/bin/env bash
# Local end-to-end check with no GPU (development; runs agents as your own user):
# a stub model endpoint drives real Copilot CLI agents through solo, solo_rules and team@2
# trials of a real ARC game, then the offline analysis runs over the results.
#   scripts/smoke_test.sh [path-to-copilot-binary]
set -euo pipefail
cd "$(dirname "$0")/.."
COPILOT="${1:-copilot}"
WORK="$(mktemp -d)"
PY=.venv/bin/python
trap 'kill $MOCK_PID 2>/dev/null || true' EXIT

$PY -m ttc.cli download --dir "$WORK/arc_envs" --games ls20
cat > "$WORK/manifest.json" <<'EOF'
{"tasks": [{"game": "ls20", "mode": "solo", "k": 1, "trial": 0},
           {"game": "ls20", "mode": "solo_rules", "k": 1, "trial": 0},
           {"game": "ls20", "mode": "team", "k": 2, "trial": 0}]}
EOF
$PY tests/mock_llm.py --port 8901 --no-models --script "arc info" "arc act 1" "arc state | head -5" \
    > "$WORK/mock.log" 2>&1 &
MOCK_PID=$!
sleep 2
OPENAI_BASE_URL=http://127.0.0.1:8901/v1 OPENAI_API_KEY=dummy \
  $PY -m ttc.cli sweep --manifest "$WORK/manifest.json" --results-root "$WORK/results" --parallel 3 -- \
  --environments-dir "$WORK/arc_envs" --work-dir "$WORK/work" --copilot-binary "$COPILOT" \
  --max-wall-seconds 600 --max-relaunches 1
$PY -m ttc.cli analyze "$WORK/results" --out "$WORK/analysis"
$PY - "$WORK/results" <<'EOF'
import json, sys, pathlib
results = [json.loads(p.read_text()) for p in pathlib.Path(sys.argv[1]).glob("*/result.json")]
assert len(results) == 3, [r["trial"]["label"] for r in results]
for r in results:
    assert r["status"] == "ok", r["error"]
    assert all(a["total_actions"] >= 1 for a in r["agents"]), r["trial"]["label"]
    assert all(a["model_calls"]["output_tokens"] > 0 for a in r["agents"]), r["trial"]["label"]
print(f"SMOKE TEST OK ({len(results)} trials) -> {sys.argv[1]}")
EOF
