"""Prompt templates.

Templates use `{name}` placeholders but also contain literal shell syntax (`$S`, `$(...)`),
so they are filled by plain string replacement rather than str.format.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files

TEAM_SECTION = """
## Teammates

You are one of {n} agents working on this game at the same time. Each agent plays its
own copy of the same game in a separate session: your actions do not affect theirs, and
an action sequence shared in notes must be replayed by each agent in its own session
(costing that agent's own actions). To keep the team in step, the game server
periodically pauses an agent that is ahead of the slowest active teammate: a paused
`arc act`/`arc click`/`arc reset` returns WAITING, is not executed, and costs nothing.
While paused, keep working: read and write notes, analyze frames, and plan.
"""


POLYOMINO_TEAM_SECTION = """
## Teammates

You are one of {n} agents working on this problem at the same time. Each agent submits
independently and every submission is scored on its own; the team's result is its best
valid submission. Submissions are judged one at a time, so yours may wait briefly in a
queue while a teammate's is being judged.
"""


def _template(name: str) -> str:
    return files(__package__).joinpath(name).read_text()


def fill(template: str, values: dict[str, object]) -> str:
    for key, value in values.items():
        template = template.replace("{" + key + "}", str(value))
    return template


@dataclass
class SharedPaths:
    """Paths as seen from inside the agent sandbox."""

    slots: str
    findings: str
    disconfirm: str
    plateau: str
    coordination: str
    base: str

    def as_dict(self) -> dict[str, str]:
        return dict(self.__dict__)


def arc_task_prompt(game: str, win_levels: int, multiplier: float, team_size: int) -> str:
    team = fill(TEAM_SECTION, {"n": team_size}) if team_size > 1 else ""
    mult = f"{multiplier:g}"
    return fill(
        _template("arc_task.md"),
        {"game": game, "win_levels": win_levels, "multiplier": mult, "team_section": team},
    )


def polyomino_task_prompt(statement: str, hours: float, team_size: int) -> str:
    team = fill(POLYOMINO_TEAM_SECTION, {"n": team_size}) if team_size > 1 else ""
    return fill(_template("polyomino_task.md"),
                {"statement": statement.strip(), "hours": f"{hours:g}", "team_section": team})


def communication_prompt(n: int, paths: SharedPaths) -> str:
    return fill(_template("communication.md"), {"n": n, "nm1": n - 1, **paths.as_dict()})


def solo_rules_prompt(paths: SharedPaths) -> str:
    return fill(_template("solo_rules.md"), paths.as_dict())
