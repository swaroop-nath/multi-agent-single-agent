# Task: polyomino packing

Write a C++17 program that solves the optimization problem below, then keep improving it
until time runs out. The run lasts about {hours} hours. Denser packings score higher.

## Problem statement

{statement}

## Submitting and scoring

Use the `submit` command (already on your PATH):

    submit <file.cpp>       compile and score a solution on the hidden test set
    submit --best           show your best valid submission so far

Add `--json` for machine-readable output.

The scorer compiles with `g++ -O2 -pipe -std=gnu++17` and runs the program on 70 hidden test
cases, with 2 seconds of CPU time and 256 MB of memory per case. For each case it computes
the ratio (total cells) / (W x H). The statement's scoring line multiplies this by 1e5; the
scorer reports the plain ratio. The packing score is the mean ratio over all 70 cases, between
0 and 1. A case that fails scores 0 and makes the whole submission invalid. A case fails on
wrong or invalid output, exceeding the time or memory limit, or a crash. Only valid
submissions count, and your result is your best valid submission.

Submit as often as you like; nothing is charged. The hidden cases are never shown: the scorer
returns only the score, validity, verdict counts and timing. To test locally, generate your
own inputs as described in the Generation section and write your own validity checks.
`g++` and `python3` (with numpy) are available.
{team_section}
## Rules

Closed book: do not use the internet, web search, curl, wget, or HTTP libraries. Do not look
for published solutions to this problem or for the scorer's test data. Use only these
instructions, your own experiments, and files in this workspace.
