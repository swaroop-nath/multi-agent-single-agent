You work this task alone. Search widely and keep improving until time runs out.

Scratch (create on first use):
- findings: {findings}
- disconfirmations: {disconfirm}
- score log: {plateau}

Your private scratch is {base}/work-0.

1. PICK AN APPROACH:
   mkdir -p {base}/work-0
   echo "<your approach + what you will deliberately not assume>" > {base}/work-0/approach

2. ALWAYS BE ACTING. Never end a turn with only prose or a plan. Every turn must
   run a command that advances or tests the work: take a real action the task
   accepts and read back its result. Recording, disconfirming, and pivoting are
   bookkeeping around real actions, never a substitute for taking one.

3. EVALUATE AND RECORD EVERY ATTEMPT. Score each change with the task's own
   scoring or feedback mechanism (find it in AGENT.md or the task instructions),
   then log the result:
   echo "[slot 0 $(date -u +%H:%M:%S)] score=<score-or-progress> family=<approach>" >> {plateau}

4. RECORD AND FALSIFY. Append concise findings with evidence and cost to
   {findings}, and label weak claims as weak. Spend part of your effort trying to
   FALSIFY your leading idea, recording negative results in {disconfirm}.

5. NEVER STOP WHILE TIME REMAINS. A working result is not the finish line; the
   clock running out is the only acceptable reason to stop. Do not declare the
   task done, final, solved, or "at the ceiling" and go idle -- a suspected
   ceiling is a claim to disconfirm, not a reason to quit.

6. BREAK PLATEAUS BY CHANGING FAMILY. You are plateaued when your best score has
   not strictly improved over 3 consecutive attempts. Then stop tuning
   and switch to a STRUCTURALLY DIFFERENT approach -- a different core principle or
   assumption, not a variant of the current one. Keep a short list of untried
   families in {base}/work-0 so you always have a next one ready.

Run a tight loop -- change -> evaluate -> record -> repeat -- without pausing. Do
not use the internet, curl, wget, HTTP libraries, or secrets.
