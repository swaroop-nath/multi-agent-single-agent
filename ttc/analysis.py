"""Offline cross-trial metrics (Sections 2.1, 3.1, 3.2 of the paper) from many result.json files.

Each trial is scored on its own (see result.json "outcome"); the comparisons need pools:

* best@k from a finite solo pool, exactly and without replacement:
      best@k = 1 - C(n - s, k) / C(n, k)
  applied per game and per level threshold, then averaged over games (equal weight).
* team@k: fraction of team trials in which any agent reached the threshold.
* Table 2: rates of reaching within d levels of a full solve.
* "Matching" pool size: smallest N with mean best@N >= team@k's final solve rate.
* Furthest level reached; for best@k the exact expected max over k-subsets of the pool.
* RHAE (Eq. 1-2): teams report best and mean agent; best@k is the expected max over k-subsets.
* Token-matched solve-rate curves from each trial's output tokens at first solve.
* Polyomino: final score = best valid submission; best@k = exact expected max of k solo trials;
  best-so-far curves over the run (mean, max, best@k), per horizon (3 h / 72 h).

    python -m ttc.cli analyze RESULTS_DIR [RESULTS_DIR ...] --out analysis/
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

SOLO_MODES = ("solo", "solo_rules")


def best_at_k(n: int, s: int, k: int) -> float:
    if n == 0 or k > n:
        return float("nan")
    return 1.0 - math.comb(n - s, k) / math.comb(n, k)


def expected_max_at_k(values: list[float], k: int) -> float:
    """E[max of a uniformly random k-subset] of a finite pool (exact)."""
    n = len(values)
    if n == 0 or k > n:
        return float("nan")
    v = sorted(values)
    # the i-th smallest (1-based) is the max with probability C(i-1, k-1) / C(n, k)
    return sum(v[i - 1] * math.comb(i - 1, k - 1) for i in range(k, n + 1)) / math.comb(n, k)


def rhae(agent: dict, baselines: list[int]) -> float:
    """Eq. 1-2 for one agent: level score min(1.15, (h/a)^2), weights w_l = l, capped by the
    weighted fraction of levels completed."""
    n = len(baselines)
    if n == 0:
        return 0.0
    w = list(range(1, n + 1))
    wsum = sum(w)
    scores = [0.0] * n
    for rec in agent.get("level_log", []):
        lvl = rec["level"]  # 1-based
        if lvl <= n:
            scores[lvl - 1] = min(1.15, (baselines[lvl - 1] / max(1, rec["level_actions"])) ** 2)
    d = agent.get("max_level", 0)
    return min(sum(w[:d]) / wsum, sum(wi * si for wi, si in zip(w, scores)) / wsum)


def load_results(paths: list[Path]) -> tuple[list[dict], list[dict], list[dict]]:
    """Returns (valid ARC trials, valid polyomino trials, infra-error trials) as flat records."""
    arc, poly, bad = [], [], []
    for root in paths:
        candidates = [root] if root.name == "result.json" else sorted(root.rglob("result.json"))
        for p in candidates:
            r = json.loads(p.read_text())
            t = r.get("trial", {})
            if r.get("status") != "ok" or "outcome" not in r:
                bad.append({"path": str(p), "label": t.get("label"), "error": r.get("error")})
                continue
            task = t.get("task", "arc")
            mode = t["mode"]
            if mode == "team" and t.get("team_prompt") in ("loose", "shared-file"):
                mode = {"loose": "team_loose", "shared-file": "team_file"}[t["team_prompt"]]
            base = {"path": str(p), "label": t["label"], "task": task, "mode": mode, "k": t["k"],
                    "tokens_total": r["totals"]["output_tokens"]}
            if task == "arc":
                arc.append({**base, "game": t["game"], "win_levels": r["game"]["win_levels"],
                            "max_level": r["outcome"]["score"], "solved": r["outcome"]["solved"],
                            "rhae_best": r["outcome"]["rhae_best_agent"],
                            "rhae_agents": [a["rhae"] for a in r["agents"]],
                            "tokens_at_solve": r["outcome"]["output_tokens_at_first_solve"]})
            else:
                poly.append({**base, "horizon_s": r["config"]["max_wall_seconds"], "score": r["outcome"]["score"],
                             "best_so_far": r["outcome"]["best_so_far"],
                             "tokens_at_best": r["outcome"]["output_tokens_at_best"]})
    return arc, poly, bad


def _nanmean(xs):
    xs = [x for x in xs if not (isinstance(x, float) and math.isnan(x))]
    return mean(xs) if xs else float("nan")


class Analysis:
    def __init__(self, results: list[dict]):
        self.results = results
        self.games = sorted({r["game"] for r in results})
        self.by = defaultdict(list)  # (mode, k, game) -> results
        for r in results:
            self.by[(r["mode"], r["k"], r["game"])].append(r)
        self.win_levels = {r["game"]: r["win_levels"] for r in results}
        self.configs = sorted({(r["mode"], r["k"]) for r in results}, key=lambda c: (c[0] != "solo", c))
        self.team_mode = "team"  # or "team_loose"; set by the caller before comparing

    def games_both(self, team_k: int, solo_mode: str) -> list[str]:
        return [g for g in self.games if self.by.get((self.team_mode, team_k, g)) and self.by.get((solo_mode, 1, g))]

    # ---- level-threshold rates ---------------------------------------------------------------------
    def solo_rate(self, mode: str, game: str, k: int, d: int) -> float:
        pool = self.by.get((mode, 1, game), [])
        target = self.win_levels[game] - d
        return best_at_k(len(pool), sum(r["max_level"] >= target for r in pool), k)

    def team_rate(self, k: int, game: str, d: int) -> float:
        trials = self.by.get((self.team_mode, k, game), [])
        if not trials:
            return float("nan")
        target = self.win_levels[game] - d
        return mean(r["max_level"] >= target for r in trials)

    def matching_pool_size(self, team_k: int, solo_mode: str = "solo") -> int | None:
        games = self.games_both(team_k, solo_mode)
        if not games:
            return None
        target = _nanmean([self.team_rate(team_k, g, 0) for g in games])
        if not target > 0:
            return None
        n_pool = min(len(self.by[(solo_mode, 1, g)]) for g in games)
        for n in range(1, n_pool + 1):
            if _nanmean([self.solo_rate(solo_mode, g, n, 0) for g in games]) >= target:
                return n
        return None

    def depth_table(self, team_k: int, solo_mode: str = "solo", max_d: int = 5) -> list[dict]:
        games = self.games_both(team_k, solo_mode)
        n_match = self.matching_pool_size(team_k, solo_mode)
        rows = []
        for d in range(max_d, -1, -1):
            row = {"levels_from_target": d,
                   f"{self.team_mode}@{team_k}": _nanmean([self.team_rate(team_k, g, d) for g in games]),
                   f"best@{team_k}": _nanmean([self.solo_rate(solo_mode, g, team_k, d) for g in games])}
            if n_match:
                row[f"best@{n_match}"] = _nanmean([self.solo_rate(solo_mode, g, n_match, d) for g in games])
            rows.append(row)
        return rows

    # ---- per-game summaries -----------------------------------------------------------------------
    def per_game(self, team_k: int, solo_mode: str = "solo") -> list[dict]:
        rows = []
        for g in self.games:
            pool = self.by.get((solo_mode, 1, g), [])
            team = self.by.get((self.team_mode, team_k, g), [])
            rows.append({
                "game": g, "levels": self.win_levels[g],
                "solo_trials": len(pool), "solo_solves": sum(r["solved"] for r in pool),
                f"best@{team_k}_solve": self.solo_rate(solo_mode, g, team_k, 0) if pool else float("nan"),
                f"{self.team_mode}@{team_k}_trials": len(team),
                f"{self.team_mode}@{team_k}_solve": mean(r["solved"] for r in team) if team else float("nan"),
                f"best@{team_k}_furthest": expected_max_at_k([r["max_level"] for r in pool], team_k),
                f"{self.team_mode}@{team_k}_furthest": mean(r["max_level"] for r in team) if team else float("nan"),
            })
        return rows

    def rhae_summary(self, team_k: int, solo_mode: str = "solo") -> dict:
        games = self.games_both(team_k, solo_mode)
        solo_best, solo_mean, team_best, team_mean = [], [], [], []
        for g in games:
            pool = [r["rhae_agents"][0] for r in self.by[(solo_mode, 1, g)]]
            solo_mean.append(mean(pool))
            solo_best.append(expected_max_at_k(pool, team_k))
            trials = self.by[(self.team_mode, team_k, g)]
            team_best.append(mean(max(r["rhae_agents"]) for r in trials))
            team_mean.append(mean(mean(r["rhae_agents"]) for r in trials))
        return {"games": len(games), "single_agent_mean": _nanmean(solo_mean),
                f"best@{team_k}": _nanmean(solo_best), f"{self.team_mode}@{team_k}_best_agent": _nanmean(team_best),
                f"{self.team_mode}@{team_k}_mean_agent": _nanmean(team_mean)}

    def token_curves(self, team_k: int, solo_mode: str = "solo", points: int = 40) -> list[dict]:
        """Solve rate vs total output-token budget; best@k splits the budget evenly over k agents."""
        games = self.games_both(team_k, solo_mode)
        totals = [r["tokens_total"] for r in self.results if r["tokens_total"]]
        if not games or not totals:
            return []
        lo, hi = max(1, min(totals) // 10), max(totals) * 2

        def solved_within(r, b):
            return r["tokens_at_solve"] is not None and r["tokens_at_solve"] <= b

        rows = []
        for i in range(points):
            budget = lo * (hi / lo) ** (i / (points - 1))
            team = [mean(solved_within(r, budget) for r in self.by[(self.team_mode, team_k, g)]) for g in games]
            best = []
            for g in games:
                pool = self.by[(solo_mode, 1, g)]
                best.append(best_at_k(len(pool), sum(solved_within(r, budget / team_k) for r in pool), team_k))
            rows.append({"total_output_tokens": round(budget), f"{self.team_mode}@{team_k}": mean(team),
                         f"best@{team_k}": _nanmean(best)})
        return rows


def _fmt(v) -> str:
    if isinstance(v, float):
        return "nan" if math.isnan(v) else f"{v:.3f}"
    return str(v)


def _print_table(title: str, rows: list[dict]) -> None:
    if not rows:
        return
    cols = list(rows[0])
    widths = [max(len(c), *(len(_fmt(r[c])) for r in rows)) for c in cols]
    print(f"\n== {title}")
    print("  ".join(c.ljust(w) for c, w in zip(cols, widths)))
    for r in rows:
        print("  ".join(_fmt(r[c]).ljust(w) for c, w in zip(cols, widths)))


def _write_csv(path: Path, rows: list[dict]) -> None:
    if rows:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


def _score_at(r: dict, t: float) -> float:
    best = 0.0
    for p in r["best_so_far"]:
        if p["t"] <= t:
            best = p["score"]
    return best


def analyze_polyomino(results: list[dict], out_dir: Path) -> dict:
    """Continuous scores: best@k is the exact expected max over k-subsets of the solo pool."""
    summary: dict = {}
    for horizon in sorted({r["horizon_s"] for r in results}):
        rs = [r for r in results if r["horizon_s"] == horizon]
        tag = f"polyomino_{horizon / 3600:g}h"
        by = defaultdict(list)
        for r in rs:
            by[(r["mode"], r["k"])].append(r)
        rows = []
        for (mode, k), group in sorted(by.items(), key=lambda kv: (kv[0][0] != "solo", kv[0])):
            scores = [r["score"] for r in group]
            rows.append({"config": f"{mode}@{k}", "trials": len(group), "mean": mean(scores),
                         "max (best run)": max(scores), "min": min(scores),
                         "mean output tokens": mean(r["tokens_total"] for r in group)})
        _print_table(f"{tag}: final score (best valid submission per trial)", rows)
        comps = []
        for (mode, k), group in sorted(by.items()):
            if not mode.startswith("team"):
                continue
            for sm in SOLO_MODES:
                pool = [r["score"] for r in by.get((sm, 1), [])]
                if not pool:
                    continue
                comps.append({"comparison": f"{mode}@{k} vs {sm}", "team@k mean": mean(r["score"] for r in group),
                              f"best@k expected ({sm})": expected_max_at_k(pool, k),
                              "team best run": max(r["score"] for r in group), "solo best run": max(pool),
                              "solo pool size": len(pool)})
        _print_table(f"{tag}: team@k vs best@k", comps)
        grid = [horizon * i / 48 for i in range(1, 49)]
        curve = []
        for t in grid:
            row = {"hours": round(t / 3600, 3)}
            for (mode, k), group in sorted(by.items()):
                vals = [_score_at(r, t) for r in group]
                row[f"{mode}@{k} mean"] = mean(vals)
                row[f"{mode}@{k} max"] = max(vals)
                if not mode.startswith("team"):
                    for kk in sorted({kk for m, kk in by if m.startswith("team")}):
                        row[f"best@{kk} from {mode}"] = expected_max_at_k(vals, kk)
            curve.append(row)
        _write_csv(out_dir / f"{tag}_final.csv", rows)
        _write_csv(out_dir / f"{tag}_comparisons.csv", comps)
        _write_csv(out_dir / f"{tag}_best_so_far_curve.csv", curve)
        summary[tag] = {"final": rows, "comparisons": comps}
    return summary


def analyze(paths: list[Path], out_dir: Path) -> dict:
    arc, poly, bad = load_results(paths)
    out_dir.mkdir(parents=True, exist_ok=True)
    if bad:
        print(f"skipping {len(bad)} infra-error trial(s): " + ", ".join(str(b["label"]) for b in bad[:10]))
    summary: dict = {"infra_errors": bad}
    if poly:
        print(f"polyomino: {len(poly)} trials")
        summary["polyomino"] = analyze_polyomino(poly, out_dir)
    if arc:
        summary["arc"] = analyze_arc(arc, out_dir)
    if not arc and not poly:
        print("no valid results found")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nwrote {out_dir}/")
    return summary


def analyze_arc(results: list[dict], out_dir: Path) -> dict:
    a = Analysis(results)
    summary: dict = {"configs": [f"{m}@{k}" for m, k in a.configs], "trials": len(results)}
    print(f"ARC: {len(results)} trials over {len(a.games)} games; configs: {', '.join(summary['configs'])}")
    for mode, k in a.configs:
        rs = [r for r in results if r["mode"] == mode and r["k"] == k]
        print(f"  {mode}@{k}: {len(rs)} trials, mean output tokens/trial "
              f"{mean(r['tokens_total'] for r in rs):,.0f}, solve rate {mean(r['solved'] for r in rs):.3f}")
    solo_modes = [m for m in SOLO_MODES if (m, 1) in a.configs]
    for team_mode, k in [(m, k) for m, k in a.configs if m.startswith("team")]:
        a.team_mode = team_mode
        for sm in solo_modes:
            tag = f"arc_{team_mode}{k}_vs_{sm}"
            depth, n_match = a.depth_table(k, sm), a.matching_pool_size(k, sm)
            per_game, rh, curves = a.per_game(k, sm), a.rhae_summary(k, sm), a.token_curves(k, sm)
            _print_table(f"{team_mode}@{k} vs {sm} pool: rate of reaching within d levels of a full solve (Table 2)", depth)
            print(f"   matching pool size for {team_mode}@{k}'s final solve rate: "
                  f"{n_match if n_match else 'not reached by the pool'}")
            _print_table(f"{team_mode}@{k} vs {sm}: per game", per_game)
            _print_table(f"{team_mode}@{k} vs {sm}: RHAE", [rh])
            _write_csv(out_dir / f"{tag}_depth.csv", depth)
            _write_csv(out_dir / f"{tag}_per_game.csv", per_game)
            _write_csv(out_dir / f"{tag}_token_curve.csv", curves)
            _plot_curve(out_dir / f"{tag}_token_curve.png", curves, k, team_mode)
            summary[tag] = {"depth": depth, "matching_pool_size": n_match, "rhae": rh, "per_game": per_game}
    return summary


def _plot_curve(path: Path, rows: list[dict], k: int, team_mode: str = "team") -> None:
    if not rows:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    x = [r["total_output_tokens"] for r in rows]
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(x, [r[f"best@{k}"] for r in rows], label=f"best@{k}")
    ax.plot(x, [r[f"{team_mode}@{k}"] for r in rows], label=f"{team_mode}@{k}")
    ax.set_xscale("log")
    ax.set_xlabel("total output tokens")
    ax.set_ylabel("games solved")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
