# ttc: test-time communication on ARC-AGI-3 and polyomino packing

A replication harness for *Scaling Discovery through Test-Time Communication*
(Park et al., arXiv 2609.21032), packaged as a self-contained eval image. The authors' repo
is empty, so this is rebuilt from the paper. It covers two of the paper's tasks:

* **ARC-AGI-3** (`--task arc`): the 25 public games, through the official `arc-agi` toolkit.
* **Frontier-CS polyomino packing** (`--task polyomino`): Frontier-CS algorithmic problem 0,
  scored on its 70 test cases with the upstream checker.

Each agent is a coding-agent CLI (`--agent-cli`):

* `copilot` (default): **GitHub Copilot CLI**, as in the paper.
* `claude-code`: **Claude Code**, e.g. with a Claude.ai subscription. Prompts, tasks, verifiers and
  results are identical, but the agent harness differs from the paper's (its own system prompt,
  tools and context management), so results are not directly comparable to the paper's numbers.
  Team-vs-solo comparisons within a run set remain fair. See "Running with Claude Code" below.

Copilot runs in BYOK ("bring your own key") mode with `COPILOT_OFFLINE=true`: no GitHub account or
Copilot subscription, and no network access apart from model calls. Two providers are supported
(`--provider`):

* `anthropic`: Claude models through the Anthropic Messages API, with an API key from the
  Anthropic Console in `ANTHROPIC_API_KEY` (a Claude.ai subscription can't be used here).
* `openai` (default): any OpenAI-compatible chat-completions endpoint, e.g. a model served with
  vLLM or SGLang, given in `OPENAI_BASE_URL` / `OPENAI_API_KEY`.

## Running a trial

| | |
|---|---|
| Self-contained, pinned image | `Dockerfile` pins the Python and Node base images by digest, hash-locks Python deps (`requirements.lock`, `agent-requirements.lock`), and installs Copilot CLI from an integrity-locked `docker/copilot/package-lock.json`. At build time it downloads the 25 ARC games (pinned and hash-checked against `ttc/arc/games.lock.json`) and the Frontier-CS files (a fixed upstream commit, hash-checked against `ttc/frontiercs/polyomino.lock.json`) |
| One trial is one command | `/app/run_trial --task arc --game G --mode {solo,solo_rules,team} --k K --trial T --results-dir /tmp/results --max-wall-seconds N`, or `--task polyomino` without `--game` (no ENTRYPOINT or CMD) |
| Model access only from env | `--provider anthropic`: reads `ANTHROPIC_API_KEY` (and optionally `ANTHROPIC_BASE_URL`, default `https://api.anthropic.com`); calls `/v1/messages`, key sent as `x-api-key`; `--model-alias` must be a real model id. `--provider openai`: reads `OPENAI_BASE_URL` (or `OPENAI_API_BASE`) and `OPENAI_API_KEY`; calls `<base>/chat/completions` with the base used exactly as given, key sent as `Bearer`. Agents never see the key |
| No other network | Copilot runs with `COPILOT_OFFLINE=true` and auto-update off; the games and test data are local |
| Results directory | `--results-dir`, default `/tmp/results` (layout below) |
| Exit codes | `0` trial ran (any score) · `2` model endpoint unreachable at start · `3` preflight failed · `4` no model call succeeded · `5` stopped by signal · `1` harness bug |
| Wall-clock limit | Stops itself at `--max-wall-seconds` minus a margin (`--wall-margin-seconds`, default max(120 s, 3%)), writes results, exits 0 with `ended_by: wall_clock` |
| Independent, rerunnable trials | No state outside the container; `--trial` is a label only |

### Team prompt (`--team-prompt`): the paper's protocol, loose, or one shared file

* `paper` (default): the Appendix A.2 communication protocol, word for word, with a pre-made
  shared folder (slots, findings, disconfirmations, score log, coordination).
* `loose`: the task prompt plus one sentence, and nothing else:

  > You are one of {n} agents working on this same task at the same time. Work as a team.

  No channel is named, no folders are pre-made, and the task prompts' teammate paragraphs are
  dropped: how (and whether) to find each other and coordinate is left to the agents. The only
  shared medium is the common working folder. Loose teams are labelled `teamloose` and analysed
  as their own configuration (`team_loose@k`), against the same solo pool.

* `shared-file`: like `loose`, plus one empty shared file the team decides how to use:

  > You are one of {n} agents working on this same task at the same time. Work as a team. There
  > is one shared file, `<task>/shared/team.md`, that every agent can read and write; you can use
  > it to communicate. As a team, figure out how best to use it.

  Only that file is pre-made. These teams are labelled `teamfile` and analysed as `team_file@k`.

  Caveat for native (non-Docker) runs: all agents run as your user, so an exploring agent can read
  its teammates' private CLI transcripts in the run folder. In Docker each agent is a separate
  user and private folders are closed.

### Facilitator (`--facilitator`): reminders to follow the paper's protocol

With `--mode team --team-prompt paper --agent-cli claude-code`, a third Claude Code agent (same
model and effort as the agents unless `--facilitator-model` / `--facilitator-effort` say otherwise)
acts as a reminder service. It is not a teammate and not a new protocol. Every
`--facilitator-interval-seconds` (default 900, first round after `--facilitator-first-seconds`) it
reads the shared folder and each agent's private scratch (read-only tools: Read, Glob, Grep), plus
each agent's own submission count and best score. It then returns one short reminder per agent, addressed by slot:
share your progress the way the protocol says (score log, findings, disconfirmations), and check the
shared files for news from your teammates. It may quote the protocol. It may not pass on content
(teammates' ideas, code or scores), add files or conventions, or give task advice. Its instructions
are in `ttc/facilitator.py` and saved per run as `facilitator/system_prompt.md`.

Agents then run with Claude Code's streaming input (`--input-format stream-json`), and each reminder
is written to the agent's stdin. Claude Code shows it at the agent's next tool call, inside the same
turn, as "The user sent a new message while you were working: [Team reminder from the harness] ...".
The agents' prompt is unchanged. Every round (prompt, inferred slots, reminders, delivery) is in
`trajectories/facilitator_rounds.jsonl.gz`, the facilitator's own session in
`trajectories/facilitator-private.tar.gz`, and its rounds and token cost in `result.json` under
`facilitator`. These teams are labelled `teamfac` and analysed as `team_fac@k`.

### Example: Claude Sonnet 4.6 on polyomino, one team@3 vs best@3, 3 hours

```bash
export ANTHROPIC_API_KEY=sk-ant-...      # an Anthropic Console API key
COMMON="--task polyomino --provider anthropic --model-alias claude-sonnet-4-6 --reasoning-effort max \
  --max-wall-seconds 10800 --price-input 3 --price-output 15 --price-cache-read 0.3 --price-cache-write 3.75"

# all four trials at once, each on its own local port; ~3 hours in total
/app/run_trial $COMMON --mode team --k 3 --trial 0 --results-dir /tmp/results/team3-0 --port 8700 &
for t in 0 1 2; do                        # best@3 = the best of three independent solo runs
  /app/run_trial $COMMON --mode solo --k 1 --trial $t --results-dir /tmp/results/solo-$t --port $((8701 + t)) &
done
wait
ttc analyze /tmp/results --out /tmp/analysis
```

Inside Docker, pass the key with `-e ANTHROPIC_API_KEY` and mount a results directory. The
`--price-*` flags (USD per million tokens) only fill in `estimated_cost_usd` in `result.json`;
check current prices before relying on them. One observation: in Anthropic mode Copilot sends
`max_tokens: 32000` regardless of `--max-output-tokens`.

### Running with Claude Code

```bash
claude setup-token                        # once: a long-lived token for your Claude subscription
export CLAUDE_CODE_OAUTH_TOKEN=...        # the token it prints
COMMON="--task polyomino --agent-cli claude-code --model-alias claude-sonnet-4-6 --reasoning-effort max \
  --max-wall-seconds 10800 --price-input 3 --price-output 15 --price-cache-read 0.3 --price-cache-write 3.75"
ttc trial $COMMON --mode team --k 3 --trial 0 --results-dir results/team3-0 --port 8700 &
for t in 0 1 2; do
  ttc trial $COMMON --mode solo --k 1 --trial $t --results-dir results/solo-$t --port $((8701 + t)) &
done
wait
```

How each agent's Claude Code is set up:
- `--safe-mode` and a fresh per-agent `CLAUDE_CONFIG_DIR`, with auto-memory and CLAUDE.md loading
  switched off: nothing from your own `~/.claude` (instructions, memories, plugins, hooks, MCP
  servers) reaches the agents.
- `showThinkingSummaries` is on, so the summarized thinking text is recorded (by default Claude
  Code keeps only an encrypted signature, with no readable text).
- Claude Code caps each response at 32k output tokens. At `--effort max`, one response can spend
  minutes thinking and hit that cap before acting; `--effort high` avoids most of this.
- Tools that would break the experiment are disallowed: web search/fetch (closed book), messaging
  other local Claude sessions (it would let separate trials talk), multi-agent workflows,
  scheduling, worktrees and remote triggers. Its subagent tool stays, like Copilot's.
- Subscription usage limits: when Claude Code reports a limit, the agent waits
  (`--limit-wait-seconds`, default 10 min) and resumes the same session; the wait counts against
  the trial's wall clock and is recorded per agent (`usage_limit_waits`).
- Tokens come from Claude Code's own transcripts, and `reported_cost_usd` is Claude Code's own
  (list-price) figure; with a subscription that is not what you pay. `--claude-auth api-key`
  (with `--provider anthropic` and `ANTHROPIC_API_KEY`) routes Claude Code through the local
  proxy instead, for exact per-call accounting, retries, and the raw model responses.
- The image does not include Claude Code yet; run it natively (below) or add it to the image.

### Running natively on a Mac (no Docker)

Fine for pilots, with three caveats: agents run as your user, so nothing *enforces* that they
stay out of the hidden test data or your files (they're told to; audit with `ttc trace`);
memory limits are checked after the fact rather than enforced; and Apple's compiler lacks
`bits/stdc++.h`, so use GCC for the judge (agents then get the same GCC as `g++`):

```bash
python3 -m venv .venv && .venv/bin/pip install -e .
brew install gcc                                    # provides g++-15 (or similar)
.venv/bin/ttc fetch-frontiercs --dir ~/ttc-data/frontiercs
# add to the trial commands:  --frontiercs-dir ~/ttc-data/frontiercs --judge-compiler g++-15
```

Task lists are in `manifests/`; each entry carries its own `max_wall_seconds`. Regenerate
with `ttc manifest NAME`.

| Manifest | Tasks | Contents | Wall clock per task |
|---|---|---|---|
| `arc_pilot` | 288 | 6 games: 16 solo + 16 solo_rules + 8 team@3 + 8 team@5 each | up to 12 h |
| `arc_full` | 4,200 | 25 games: 64 + 64 + 20 + 20 each (the paper's counts) | up to 12 h |
| `polyomino_pilot` | 28 | 12 solo + 12 solo_rules + 4 team@3 | 3 h |
| `polyomino_full` | 166 | 60 + 60 + 20 team@3 at 3 h; 12 + 12 + 2 team@4 at 72 h (the paper's counts) | 3 h / 72 h |

**Resources:** 4 CPU / 8 GiB per trial for every mode. That is the paper's Table 4 for ARC:
one container shared by the whole team. The paper gave polyomino 2 CPU / 6 GiB but ran its
scorer as a separate service; here the scorer runs inside the container, so it gets 2 more
CPUs. Copilot uses about 0.3–0.6 GiB per agent, so k ≤ 5 fits; scale memory for larger teams.

### How long a run takes

* **Polyomino** trials run for a fixed time, matching the paper: 3 h (Frontier-CS's limit) or
  72 h. The pilot is 84 task-hours; the full manifest is 420 + 1,872 = 2,292 task-hours.
* **ARC** has no fixed time limit in the paper. A trial ends when every agent has won,
  exhausted a level's action budget, or hit `--max-wall-seconds`. Budgets are 5× the human
  baseline per level, so one agent's cap for a whole game is 855–9,215 actions (median 3,190).
  Agents usually spend at least one model call per action, so with model calls taking about
  a minute, strong agents on long games will reach the 12 h cap before exhausting their
  budget; weak agents exhaust an early level's budget much sooner. Upper bounds: pilot ≤ 3,456
  task-hours, full ≤ 50,400. Check `ended_by: wall_clock` in results to see how often the cap
  cut runs short.

## The verifiers

**ARC-AGI-3: the game engine itself.** The official toolkit runs the pinned game code, and a
level counts as cleared when the engine says so.
- The trial's `score` is the number of levels cleared by the best agent; `solved` means it
  cleared them all.
- RHAE uses the official formula with the official per-level human baselines.
- Agents see the same signal live (`arc info` shows `levels_completed`). This is the dense
  "agent-accessible verifier" the paper relies on. There's no hidden data: what agents see is
  what's scored.

**Polyomino: a re-implementation of the Frontier-CS judge** (`ttc/frontiercs/judge.py`).
- It uses the 70 test cases and the unmodified upstream checker (`chk.cc` with `testlib.h`),
  and matches the official judge's compile flags (`g++ -O2 -pipe -std=gnu++17`), limits (2 s
  CPU, 4 s wall, 256 MB per case), verdict rules (the checker must exit 0 or 7) and scoring.
  - A case's score is its `Ratio:`, which is cells / (W×H).
  - A submission's score is the mean over the 70 cases.
  - The official leaderboard number is 100× this mean.
- As in the paper and the problem statement, a submission with any failed case is *invalid*.
- A trial's score is its **best valid submission**.
- Agents submit as often as they like and get back only the aggregate score, validity,
  verdict counts and timing. Per-case results stay in the harness ledger.
- Sanity check: the human-best reference solution in the Frontier-CS repo scores **0.891**
  here. The paper reports 0.894 for the prior best and 0.893 for its best single agent.

Differences from the official judge:
- **Compiler:** GCC 12 (Debian bookworm) instead of 11 (Ubuntu 22.04).
- **Limits:** enforced with rlimits and `wait4` rusage instead of go-judge's cgroups.
- **CPU contention:** the judge shares the container's CPUs with the agents. A solution that
  budgets itself by wall-clock time can score slightly differently from run to run; this also
  happens in the official judge.

### Standalone verifiers (`ttc/verifiers/`)

The verifiers can also run on their own, to re-score finished trials without trusting their
`result.json`, or to score a submission directly:

```bash
/app/verify_trial RESULTS_DIR [...] [--out report.json]   # exit 0 only if every trial matches
ttc score-polyomino solution.cpp [--per-case]            # judge any C++17 solution on the 70 cases
ttc score-arc --game lp85 --actions actions.jsonl        # replay any action sequence: {"action", "x"?, "y"?}
```

* **ARC** (`verifiers/arc.py`) replays every agent's recorded actions
  (`trajectories/game_events.jsonl.gz`) from a fresh copy of the pinned game in the official
  engine.
  - It recomputes levels cleared, actions per level, win or exhaustion, RHAE and the score.
  - Each recorded action carries a hash of the frame the live server returned, so the replay
    also confirms, step by step, that agents saw exactly what the engine produces.
  - It uses the engine and the budget rule only, not the live game server.
* **Polyomino** (`verifiers/polyomino.py`) re-judges `best_solution.cpp` and checks that it's
  valid and within `--tolerance` (default 0.005) of the recorded score. The tolerance is needed
  because solutions that budget by wall-clock time vary slightly between runs.
  - As root, it runs the program as `ttc-judge`; untrusted code never runs as root.

Every trial also verifies itself before exiting and records the outcome in `result.json` under
`verification`:
- `status` is `match`, `mismatch` (with the differences listed), `skipped` (not enough wall
  clock left) or `error`.
- A mismatch doesn't change the exit code; the score stands and the mismatch is there for
  whoever audits the results.

## What the harness adds around Copilot

* **Game server** (`ttc/arc/game.py`). It gives each agent its own ARC session over the
  official `arc-agi` toolkit, run offline.
  - Per-level budget of 5× the human baseline; an agent that runs out is ended.
  - The team sync barrier from Appendix A.3: a refused action is not charged, and finished,
    exhausted or idle agents don't block the others.
* **Model proxy** (`ttc/services.py`, on 127.0.0.1). Agents call it, never the endpoint.
  - It holds the real key and attributes every call to an agent, counting cache reads and
    writes separately.
  - It retries 429/5xx/529/transport errors with backoff, including Anthropic overload errors
    that arrive inside an HTTP 200 stream.
  - It sends SSE keepalive comments while a call is slow.
  - OpenAI mode: it adds `max_tokens` (Copilot never sends it) and maps `developer` to
    `system`. Both modes: it answers `/models` locally.
  - It detects context overflow (an error saying the prompt is too long, or an empty HTTP 200
    with `finish_reason: "length"`) and repeated failures, and ends that agent with
    `context_overflow` or `model_errors`.
* **Polyomino scorer** (`ttc/tasks/polyomino.py`, `ttc/frontiercs/`):
  - a `submit` command for agents;
  - a queue, so submissions are judged one at a time with cases in parallel
    (`--judge-parallelism`);
  - a ledger of every submission and its source;
  - the best-so-far timeline;
  - `best_solution.cpp` in the results.
* **Isolation.** The harness runs as root; agent *i* runs as the unprivileged user
  `ttc-agent<i>` (via `setpriv`).
  - Agents get a scrubbed environment built from an allowlist, with no API key.
  - Game source is root-only (`0700`).
  - Polyomino test data is root-only as well.
  - Judged programs run as a separate user, `ttc-judge`. Their test input comes on stdin and
    their output goes to a pipe. `RLIMIT_FSIZE=0` means they can't write files, and `/tmp`,
    `/var/tmp` and `/dev/shm` are closed to them, so they can't copy hidden test inputs where
    agents could read them.
  - Preflight checks confirm that an agent can reach its task server but cannot read the game
    files or test data.
* **Prompts** (`ttc/prompts/`):
  - `communication.md` is Appendix A.2 word for word (team mode).
  - `solo_rules.md` holds the A.2 rules that don't involve teammates (`solo_rules` mode). It
    controls for the fact that the paper gives those rules only to teams.
  - `arc_task.md` and `polyomino_task.md` are the task descriptions. The paper doesn't publish
    its own. The polyomino one embeds the Frontier-CS statement word for word and explains
    the scorer.

## Results directory

```
result.json                     schema_version 2 (see below)
best_solution.cpp               polyomino: the best valid submission
trajectories/
  prompt.md                     the exact prompt every agent received
  agent-<i>.events.jsonl.gz     the agent CLI's event stream: the model's reasoning, messages,
                                every tool call and its full result
  agent-<i>-private.tar.gz      the agent's private state: the CLI's session store (Copilot's, or
                                Claude Code's full transcripts incl. subagents), its home and
                                temp dirs, logs
  model_responses.jsonl.gz      every raw model response the proxy relayed, thinking included
                                (--log-model-responses, on by default; empty for Claude Code with
                                a subscription, whose calls bypass the proxy: its transcripts in
                                agent-<i>-private.tar.gz have the same content)
  model_calls.jsonl.gz          every model call: agent, status, retries, latency, finish_reason, tokens
  shared_workspace.tar.gz       the shared folder (findings, disconfirmations, slots, score log,
                                coordination) and every agent's scratch/work-<slot> dir
  game_events.jsonl.gz          ARC: every game action, refusal and session end
  submissions.jsonl.gz          polyomino: every submission, with per-case verdicts and ratios
  submission_sources.tar.gz     polyomino: every submitted source file
```

To read a run as a timeline (thinking, messages, tool calls and results, per agent):

```bash
ttc trace RESULTS_DIR [--agent 0] [--max-chars 0]
```

With Claude models the thinking is Anthropic's summarized thinking (the raw chain of thought is
never returned by the API). Program caches (e.g. Copilot's unpacked runtime) are left out of the
private archives.

`result.json` (schema 2) contains:
- `trial`: task, mode, k, trial, plus the game and pinned `game_id` (ARC) or the Frontier-CS
  commit (polyomino).
- `status` (`ok`/`infra_error`), `exit_code`, `error`.
- `outcome`, for ARC:
  - `score` is the levels cleared by the best agent;
  - also `max_score`, `normalized_score`, `solved`, `rhae_best_agent`, `rhae_mean_agent`,
    `first_solve_seconds`, `output_tokens_at_first_solve`.
- `outcome`, for polyomino:
  - `score` is the best valid submission's mean ratio;
  - also `best_submission`, `best_score_any_submission`, submission counts,
    `time_to_best_seconds`, `output_tokens_at_best`, and the `best_so_far` timeline.
- `agents[]`, one entry per agent:
  - `termination_reason`: `finished`, `budget_exhausted`, `wall_clock`, `context_overflow`,
    `model_errors`, `agent_exited`, `teammate_won` or `trial_ended`;
  - levels, actions per level, level completion log, RHAE;
  - model-call counts and tokens;
  - Copilot exit codes and relaunches.
- `totals` (tokens incl. cache reads/writes, and `estimated_cost_usd` when `--price-*` is given),
  and `degradation`: model-call failure rate, retries, overflows, crashes.
- `sampling_params_sent`: the sampling parameters Copilot sent (a server may override them).
- `versions`: harness, Copilot CLI, arc-agi, build commit.
- `config`: every setting, with no secrets.
- `verification`: the trial's self-check with the standalone verifier (see above).

## Cross-trial metrics (offline)

```bash
ttc analyze path/to/collected/results/ --out analysis/
```

It finds every `result.json` under the given paths, skips `infra_error` trials, and reports:
- exact best@k from the solo pool, 1 − C(n−s,k)/C(n,k), per level threshold and averaged
  over games;
- team@k, and the depth table (Table 2);
- the number of independent agents needed to match team@k;
- furthest level reached, and RHAE (best and mean agent);
- token-matched solve-rate curves.

For polyomino, per horizon (3 h or 72 h), it reports:
- final scores per config;
- team@k mean against best@k (the exact expected maximum over k-trial subsets of the solo
  pool);
- best team run against best solo run (the paper's figures);
- best-so-far curves over the run.

Each team is compared against both the `solo` and the `solo_rules` pool.

## Build and test

```bash
docker build -t ttc-arc --build-arg GIT_COMMIT=$(git rev-parse HEAD) .
docker run --rm ttc-arc /app/scripts/container_selftest.sh    # ARC + polyomino isolation and stub end-to-end, offline

# development without Docker (agents run as your user: no isolation)
python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'
.venv/bin/pytest                                              # judge tests need the pinned files:
.venv/bin/ttc fetch-frontiercs --dir /tmp/fcs && TTC_FRONTIERCS_DIR=/tmp/fcs .venv/bin/pytest
scripts/smoke_test.sh "$(which copilot)"                      # stub endpoint, 3 ARC trials, then analyze
```

`tests/mock_llm.py` is a stub endpoint for testing without a GPU or API key; it speaks OpenAI
chat completions, or the Anthropic Messages API with `--provider anthropic`. It can
also imitate awkward servers: send the whole response at once after a long delay, put all tool
calls in a single delta, return `finish_reason: "length"` overflows, return 503s, and refuse
`/models`.

### What testing Copilot CLI 1.0.91 against the stub showed
1. It survives a streamed call whose first byte arrives after **5 minutes** with no keepalive.
   The proxy's keepalives (after 20 s, every 15 s) are extra margin and were tested with
   40-second calls.
2. It accepts all tool calls in one delta.
3. It sends only `system`/`user`/`assistant`/`tool` roles, even when a well-known model id is set,
   and never calls `/models`. It sends `temperature`/`top_p` (or `reasoning_effort` when set)
   and never sends `max_tokens`.
4. On an empty `finish_reason: "length"` response it retries twice, then **exits 0 silently**.
   The proxy detects this, so the harness ends the agent with `context_overflow` instead of
   relaunching it into the same context.

## Choices the paper doesn't specify

- **Model and CLI version:** the paper used Sonnet 4.6 with Copilot CLI 1.0.54; this image
  pins 1.0.91, which has BYOK.
- **Restarts:** if Copilot exits while the agent's game is still live, the same session is
  resumed with a neutral continue message, identically in every mode (`--max-relaunches`).
- **Budget bookkeeping:** each level's budget is a cumulative counter per level index, so a
  full reset can't buy fresh budget. RESET counts as an action (`--count-reset-as-action`).
- **Barrier idle timeout:** 20 minutes (`--idle-timeout-seconds`).
- **Subagents:** Copilot gives each agent `task`/`write_agent` tools, so an agent can spawn
  helpers. Their calls go through the same proxy and count against that agent. To disable
  them, pass `--copilot-extra-arg=--excluded-tools --copilot-extra-arg=task` (and the same
  for `read_agent`, `list_agents`, `write_agent`).
- **Game seed:** every trial uses seed 0, so all trials play the same game, as in the
  benchmark (`--game-seed`).
