"""Tasks the harness can run: ARC-AGI-3 (`arc`) and Frontier-CS polyomino packing (`polyomino`)."""

from __future__ import annotations


def make_task(settings, launcher, work):
    if settings.task == "arc":
        from .arc import ArcTask
        return ArcTask(settings, launcher, work)
    if settings.task == "polyomino":
        from .polyomino import PolyominoTask
        return PolyominoTask(settings, launcher, work)
    raise ValueError(settings.task)
