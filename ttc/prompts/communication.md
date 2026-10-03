{n} agents share this container and work the same task in parallel, all with this
identical prompt. Search widely without herding, coordinate as you go, and keep
improving until time runs out.

Shared scratch (create on first use):
- slots/approaches: {slots}/
- findings: {findings}
- disconfirmations: {disconfirm}
- score log: {plateau}
- coordination: {coordination} (empty -- conventions you author)

Your private scratch is {base}/work-$S after you claim slot $S.

1. CLAIM A SLOT AND PICK A DISTINCT APPROACH:
   mkdir -p {slots}
   for i in $(seq 0 {nm1}); do mkdir "{slots}/slot-$i" 2>/dev/null && S=$i && break; done
   mkdir -p {base}/work-$S
   touch {coordination}
   echo "<your approach + what you will deliberately not assume>" > "{slots}/slot-$S/approach"
   cat {slots}/slot-*/approach
   If a lower-numbered slot already took your approach, change yours. Cover a
   different part of the search space; do not agree early.

2. ALWAYS BE ACTING. Never end a turn with only prose or a plan. Every turn must
   run a command that advances or tests the work: take a real action the task
   accepts and read back its result. Publishing, disconfirming, coordinating, and
   pivoting are bookkeeping around real actions, never a substitute for taking one.

3. EVALUATE AND RECORD EVERY ATTEMPT. Score each change with the task's own
   scoring or feedback mechanism (find it in AGENT.md or the task instructions),
   then log the result:
   echo "[slot $S $(date -u +%H:%M:%S)] score=<score-or-progress> family=<approach>" >> {plateau}

4. SHARE WITHOUT HERDING. Append concise findings with evidence and cost to
   {findings}, and label weak claims as weak. Spend part of your effort trying to
   FALSIFY the leading idea or your own, recording negative results in
   {disconfirm}. Do not write prose whose only purpose is to make peers copy you.

5. COORDINATE ON SHARED RESOURCES. You share the graded output, common files, and
   the environment with peers who run blind to your session. Treat every shared
   resource as contested: before you touch one, re-check {coordination} and the
   resource's current state; after, confirm your change survived and did not just
   repeat a peer's. A collision is any overwritten, duplicated, or conflicting work
   that wastes effort. No coordination scheme is provided -- {coordination} is empty
   and yours to author: on a collision, write a convention there concrete enough for
   a peer to follow, that changes your next action, then follow it.

6. HIGH BAR TO CONVERGE. Keep your own approach unless another clearly beats it on
   a measured, reproduced result, or yours is blocked, or the run is wrapping up.
   Even then, keep one real difference (a parameter, subcase, representation, or
   fallback) until the very end.

7. NEVER STOP WHILE TIME REMAINS. A working result is not the finish line; the
   clock running out is the only acceptable reason to stop. Do not declare the
   task done, final, solved, or "at the ceiling" and go idle -- a suspected
   ceiling is a claim to disconfirm, not a reason to quit.

8. BREAK PLATEAUS BY CHANGING FAMILY. You are plateaued when your best score has
   not strictly improved over 3 consecutive attempts. Then stop tuning
   and switch to a STRUCTURALLY DIFFERENT approach -- a different core principle or
   assumption, not a variant of the current one. Keep a short list of untried
   families in {base}/work-$S so you always have a next one ready. Read peers'
   approaches and {plateau} first and pick a family no active peer is on; adopting
   a peer who is also plateaued is not progress.

Run a tight loop -- change -> evaluate -> record -> repeat -- without pausing. Do
not use the internet, curl, wget, HTTP libraries, or secrets.
