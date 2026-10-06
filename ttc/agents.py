"""Launching agent processes.

All k agents of a trial run in this container (the paper's shared "task container"). When the
harness runs as root, agent i runs as the unprivileged user `<prefix><i>` (primary group
`<group>`), so agents cannot read the game source (root-only, mode 0700), the harness's
environment (which holds the API key), or each other's private state. Agents always get a
scrubbed environment built from an allowlist, never the harness's own environment.
"""

from __future__ import annotations

import asyncio
import grp
import os
import pwd
import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path

UMASK_WRAPPER = ["/bin/sh", "-c", 'umask 002; exec "$@"', "sh"]


@dataclass
class Identity:
    user: str
    uid: int
    gid: int


class AgentLauncher:
    def __init__(self, drop_privileges: bool, user_prefix: str, group: str, k: int):
        self.drop = drop_privileges
        self.ids: list[Identity | None] = [None] * k
        if self.drop:
            gid = grp.getgrnam(group).gr_gid
            for i in range(k):
                name = f"{user_prefix}{i}"
                try:
                    self.ids[i] = Identity(name, pwd.getpwnam(name).pw_uid, gid)
                except KeyError:
                    raise SystemExit(f"agent user {name} does not exist (the image creates "
                                     f"{user_prefix}0..N; raise N for larger teams)")

    @staticmethod
    def should_drop(mode: str) -> bool:
        return mode == "auto" and os.geteuid() == 0

    def argv(self, i: int | None, argv: list[str]) -> list[str]:
        """i=None: run as the harness user (the facilitator), not as an agent."""
        ident = self.ids[i] if i is not None else None
        if ident is None:
            return UMASK_WRAPPER + argv
        return ["setpriv", f"--reuid={ident.uid}", f"--regid={ident.gid}", "--clear-groups", "--",
                *UMASK_WRAPPER, *argv]

    def chown(self, i: int, path: Path) -> None:
        ident = self.ids[i]
        if ident is None:
            return
        for p in [path, *path.rglob("*")]:
            os.chown(p, ident.uid, ident.gid)

    def chown_group(self, path: Path) -> None:
        """Shared tree: writable by every agent (setgid dirs so new files keep the group)."""
        if not self.drop:
            return
        gid = self.ids[0].gid
        for p in [path, *path.rglob("*")]:
            os.chown(p, 0, gid)
            if p.is_dir():
                os.chmod(p, 0o2775)
            else:
                os.chmod(p, 0o775 if p.stat().st_mode & 0o111 else 0o664)

    async def spawn(self, i: int, argv: list[str], env: dict[str, str], cwd: str,
                    stdout_path: Path, stdin_pipe: bool = False) -> asyncio.subprocess.Process:
        out = open(stdout_path, "ab")
        try:
            return await asyncio.create_subprocess_exec(
                *self.argv(i, argv), env=env, cwd=cwd, stdout=out, stderr=asyncio.subprocess.STDOUT,
                stdin=asyncio.subprocess.PIPE if stdin_pipe else asyncio.subprocess.DEVNULL,
                start_new_session=True)
        finally:
            out.close()

    async def run(self, i: int | None, argv: list[str], env: dict[str, str], cwd: str,
                  timeout: float = 60) -> tuple[int, str]:
        p = await asyncio.create_subprocess_exec(*self.argv(i, argv), env=env, cwd=cwd,
                                                 stdout=asyncio.subprocess.PIPE,
                                                 stderr=asyncio.subprocess.STDOUT,
                                                 stdin=asyncio.subprocess.DEVNULL)
        try:
            out, _ = await asyncio.wait_for(p.communicate(), timeout)
        except asyncio.TimeoutError:
            p.kill()
            await p.wait()
            return -1, "timeout"
        return p.returncode, out.decode(errors="replace")

    @staticmethod
    def _descendants(pid: int) -> list[int]:
        """All descendant pids (a CLI may run tool commands in their own process groups)."""
        try:
            out = subprocess.run(["ps", "-A", "-o", "pid=,ppid="], capture_output=True, text=True, timeout=10).stdout
        except (OSError, subprocess.SubprocessError):
            return []
        children: dict[int, list[int]] = {}
        for line in out.splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                children.setdefault(int(parts[1]), []).append(int(parts[0]))
        found, stack = [], [pid]
        while stack:
            for c in children.get(stack.pop(), []):
                found.append(c)
                stack.append(c)
        return found

    @classmethod
    async def kill(cls, proc: asyncio.subprocess.Process) -> None:
        if proc.returncode is not None:
            return
        tree = cls._descendants(proc.pid)  # collect before the parent dies and children get re-parented
        for sig in (signal.SIGTERM, signal.SIGKILL):
            for target in (proc.pid, *tree):
                try:
                    if target == proc.pid:
                        os.killpg(proc.pid, sig)
                    else:
                        os.kill(target, sig)
                except (ProcessLookupError, PermissionError):
                    pass
            try:
                await asyncio.wait_for(proc.wait(), 10)
                if sig == signal.SIGTERM:
                    for target in tree:  # make sure detached children went too
                        try:
                            os.kill(target, signal.SIGKILL)
                        except (ProcessLookupError, PermissionError):
                            pass
                return
            except asyncio.TimeoutError:
                continue

    async def kill_all_owned(self) -> None:
        """Kill anything still running as an agent user (e.g. shells Copilot detached)."""
        for ident in self.ids:
            if ident is not None:
                p = await asyncio.create_subprocess_exec("pkill", "-KILL", "-u", ident.user,
                                                         stdout=asyncio.subprocess.DEVNULL,
                                                         stderr=asyncio.subprocess.DEVNULL)
                await p.wait()
