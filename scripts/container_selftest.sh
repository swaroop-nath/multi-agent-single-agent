#!/bin/sh
# In-container self-test (run as root inside the image, no network needed):
#   docker run --rm ttc-arc /app/scripts/container_selftest.sh
# Starts stub model endpoints and runs one ARC team@2 trial and one polyomino team@2 trial
# through /app/run_trial, then checks isolation: agents ran as unprivileged users, never saw the
# API key, could not read game source or test data, and judged programs could not write files.
set -eu
WORK=$(mktemp -d)
chmod 755 "$WORK"  # agents (unprivileged) must be able to traverse to their workspace
PY=/app/.venv/bin/python
STUB=/app/tests/mock_llm.py
FIX=/app/tests/fixtures

$PY $STUB --port 18901 --first-byte-delay 25 --script \
    "id -un >> shared/whoami" \
    "env | grep -c OPENAI >> shared/env_check || echo 0 >> shared/env_check" \
    "ls /opt/arc_envs >> shared/ls_check 2>&1 || echo DENIED >> shared/ls_check" \
    "arc info" "arc state | head -3" > "$WORK/stub_arc.log" 2>&1 &
STUB_ARC=$!
$PY $STUB --port 18902 --script \
    "cat /opt/frontiercs/algorithmic/problems/0/testdata/1.in >> shared/data_check 2>&1 || echo DENIED >> shared/data_check" \
    "cp $FIX/leak_attempt.cpp scratch/leak.cpp && submit scratch/leak.cpp" \
    "cp $FIX/simple_pack.cpp scratch/pack.cpp && submit scratch/pack.cpp" > "$WORK/stub_poly.log" 2>&1 &
STUB_POLY=$!
trap 'kill $STUB_ARC $STUB_POLY 2>/dev/null || true' EXIT
sleep 2

rc_arc=0
OPENAI_BASE_URL=http://127.0.0.1:18901/v1 OPENAI_API_KEY=selftest-secret-key \
  /app/run_trial --task arc --game lp85 --mode team --k 2 --trial 0 --results-dir "$WORK/arc" \
  --work-dir "$WORK/work" --max-wall-seconds 600 --max-relaunches 0 --port 18700 || rc_arc=$?
rc_poly=0
OPENAI_BASE_URL=http://127.0.0.1:18902/v1 OPENAI_API_KEY=selftest-secret-key \
  /app/run_trial --task polyomino --mode team --k 2 --trial 0 --results-dir "$WORK/poly" \
  --work-dir "$WORK/work" --max-wall-seconds 600 --max-relaunches 0 --port 18710 || rc_poly=$?

$PY - "$WORK" "$rc_arc" "$rc_poly" <<'EOF'
import json, os, sys, pathlib
work, rc_arc, rc_poly = pathlib.Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
arc_shared = work / "work/lp85-team2-000/task/shared"
poly_shared = work / "work/polyomino-team2-000/task/shared"
read = lambda p: p.read_text() if p.exists() else ""
arc = json.loads((work / "arc/result.json").read_text())
poly = json.loads((work / "poly/result.json").read_text())
leaks = [p for p in ("/tmp/ttc_leak.txt", "/var/tmp/ttc_leak.txt", "/dev/shm/ttc_leak.txt")
         if os.path.exists(p)] + [str(p) for p in (work / "work").rglob("ttc_leak.txt") if p.stat().st_size]
checks = {
    "arc: exit 0, status ok": rc_arc == 0 and arc["status"] == "ok",
    "arc: game source hidden (preflight)": arc["preflight"].get("game_source_hidden_from_agents") is True,
    "arc: agents ran as ttc-agent users": sorted(set(read(arc_shared / "whoami").split())) == ["ttc-agent0", "ttc-agent1"],
    "arc: no OPENAI_* vars in agent env": set(read(arc_shared / "env_check").split()) == {"0"},
    "arc: agents cannot list /opt/arc_envs": any(s in read(arc_shared / "ls_check") for s in ("Permission denied", "DENIED")),
    "arc: keepalive path (25 s calls) worked": all(a["model_calls"]["ok"] >= 5 for a in arc["agents"]),
    "arc: trajectories written": (work / "arc/trajectories/agent-0.events.jsonl.gz").exists(),
    "poly: exit 0, status ok": rc_poly == 0 and poly["status"] == "ok",
    "poly: test data hidden (preflight)": poly["preflight"].get("test_data_hidden_from_agents") is True,
    "poly: agents cannot read test data": any(s in read(poly_shared / "data_check") for s in ("Permission denied", "DENIED")),
    "poly: judged program could not write files": not leaks,
    "poly: valid submission scored": poly["outcome"]["valid_submissions"] >= 1 and 0.2 < poly["outcome"]["score"] < 0.4,
    "poly: best_solution.cpp written": (work / "poly/best_solution.cpp").exists(),
    "arc: self-verification matched (replay)": arc.get("verification", {}).get("status") == "match",
    "poly: self-verification matched (re-judge)": poly.get("verification", {}).get("status") == "match",
}
for name, ok in checks.items():
    print(("PASS " if ok else "FAIL ") + name)
if leaks:
    print("leaked:", leaks)
sys.exit(0 if all(checks.values()) else 1)
EOF
