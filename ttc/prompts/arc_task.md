# Task: ARC-AGI-3 game `{game}`

You are playing an unfamiliar turn-based grid-world game. There are no instructions:
infer the rules, the goal, and the effect of each control from play alone, then clear
as many levels as possible. The game has {win_levels} levels; clearing all of them wins.
What you learn on earlier levels usually carries over to later ones.

## Interface

Interact with the game only through the `arc` command (already on your PATH):

    arc info                game status: level, levels to win, available actions, budget left
    arc state               print the current frame (free; takes no action)
    arc act <n>             take simple action n (one of the available actions 1-5, 7)
    arc click <x> <y>       take action 6 at column x, row y (0-63), if available
    arc reset               take RESET (restarts the current attempt; counted as an action)

Add `--json` to any command for machine-readable output (full frame data) that you can
parse with Python, and `--all-frames` to see every animation frame an action produced.
Frames are 64x64 grids. Each cell is a color index 0-15 printed as one hex digit
(0-9, a-f); row 0 is the top and column 0 is the left.

## Action budget and scoring

Each level has its own action budget: {multiplier}x the human baseline for that level.
Only game actions (`act`, `click`, `reset`) are charged. Reading the state, thinking,
writing notes, and running code are free. Unused budget does not carry over to the next
level. If you use up a level's budget without clearing it, your game session ends.
Spend actions deliberately: form hypotheses and test them with as few actions as you can.

Your score is the number of levels cleared (`levels_completed` in `arc info`). Use it as
this task's scoring and feedback mechanism.
{team_section}
## Rules

Closed book: do not use the internet, web search, curl, wget, or HTTP libraries, and do
not look for public ARC solutions, replays, pages, or game source code. Use only these
instructions, your own observations, and files in this workspace.
