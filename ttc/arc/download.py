"""Pinned ARC-AGI-3 games.

`games.lock.json` (shipped in this package) records, for each public game, the exact version,
level baselines, and the SHA-256 of its source. The image build downloads exactly those
versions (`ttc download`), and every trial re-verifies the hashes before it starts, so all
trials play identical game code with no network access.
"""

from __future__ import annotations

import hashlib
import json
import logging
from importlib.resources import files
from pathlib import Path

log = logging.getLogger("ttc")


def default_lock_path() -> Path:
    return Path(str(files("ttc.arc").joinpath("games.lock.json")))


def load_lock(path: str | Path | None = None) -> dict:
    return json.loads(Path(path or default_lock_path()).read_text())


def _source_hash(game_dir: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(game_dir.glob("*.py")):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def game_dir(environments_dir: str | Path, game_id: str) -> Path:
    base, version = game_id.split("-", 1)
    return Path(environments_dir) / base / version


def write_lock(environments_dir: str, out_path: str | Path) -> dict:
    """Download every public game's current version and record it (maintainers only)."""
    from arc_agi import Arcade, OperationMode

    arcade = Arcade(operation_mode=OperationMode.NORMAL, environments_dir=environments_dir)
    games = {}
    for e in sorted(arcade.get_environments(), key=lambda e: e.game_id):
        if arcade.make(e.game_id) is None:
            raise SystemExit(f"failed to download {e.game_id}")
        base = e.game_id.split("-", 1)[0]
        games[base] = {"game_id": e.game_id, "levels": len(e.baseline_actions or []),
                       "baseline_actions": list(e.baseline_actions or []),
                       "source_sha256": _source_hash(game_dir(environments_dir, e.game_id))}
    lock = {"source": "ARC-AGI-3 public games via the arc-agi toolkit", "games": games}
    Path(out_path).write_text(json.dumps(lock, indent=2) + "\n")
    return lock


def download_locked(environments_dir: str, lock_path: str | Path | None = None,
                    games: list[str] | None = None) -> list[str]:
    from arc_agi import Arcade, OperationMode

    lock = load_lock(lock_path)["games"]
    wanted = games or sorted(lock)
    arcade = Arcade(operation_mode=OperationMode.NORMAL, environments_dir=environments_dir)
    for g in wanted:
        entry = lock[g]
        if arcade.make(entry["game_id"]) is None:
            raise SystemExit(f"failed to download {entry['game_id']}")
        log.info("cached %s", entry["game_id"])
    problems = verify(environments_dir, wanted, lock_path)
    if problems:
        raise SystemExit("downloaded games do not match the lock:\n  " + "\n  ".join(problems))
    return wanted


def verify(environments_dir: str, games: list[str], lock_path: str | Path | None = None) -> list[str]:
    lock = load_lock(lock_path)["games"]
    problems = []
    for g in games:
        entry = lock.get(g)
        if entry is None:
            problems.append(f"{g}: not in the games lock")
            continue
        d = game_dir(environments_dir, entry["game_id"])
        if not d.is_dir():
            problems.append(f"{g}: {d} missing")
            continue
        got = _source_hash(d)
        if got != entry["source_sha256"]:
            problems.append(f"{g}: source hash {got[:12]} != locked {entry['source_sha256'][:12]}")
        meta = json.loads((d / "metadata.json").read_text())
        if list(meta.get("baseline_actions") or []) != entry["baseline_actions"]:
            problems.append(f"{g}: baselines differ from the lock")
    return problems
