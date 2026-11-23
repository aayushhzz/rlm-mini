# Paper notes: Prime Agent, a self-improving RLM harness

Karten, Zhang, et al. (Prime Intellect, Princeton, MIT), arXiv 2608.23552, August 2026.
PDF: `paper/prime-agent.pdf`.

## The mechanism in my own words

A language model can only use what is in its weights and its current context window. Prime Agent treats
everything else as addressable memory outside the model, in four levels: weights (L0), active context (L1),
a live Python kernel with recursive sub-agents (L2), and disk-backed history, memories, skills and prompts
(L3). The harness's job is to move information between these levels cheaply, and to let the model decide
how.

1. **The REPL is the only tool.** Each session owns a persistent IPython kernel. The model acts by writing
   Python cells. Values persist across turns, so large inputs (logs, specs, evaluator output) live as
   variables and are never re-serialized into the prompt. Only what a cell prints comes back.
2. **Recursive Language Model (RLM) calls are functions.** Inside the kernel the model can call a model:
   `llm()` for a cheap single call, `rlm()` for a full child agent with its own context, kernel and history.
   In the paper `rlm` is asynchronous: it returns a handle at once and children report back through
   agent-to-agent message queues. The model chooses between local code, sequential delegation and parallel
   children; the harness fixes only the execution semantics, not a workflow graph.
3. **History is append-only.** Model calls, tool calls, messages and harness edits are recorded as events.
   Compaction replaces a prefix of L1 with a summary but keeps the originals on disk for REPL retrieval.
   Token accounting sums the root and all descendants, so delegation is never free in the reported cost.
4. **Continual Harness.** Typed, versioned state that supplements the base prompt: prompt notes (rules),
   memories (facts), skills (executable procedures) and subagent specs (roles). It supports CRUD. `/refine`
   runs a model over trajectory evidence and turns it into an edit, applied at a turn boundary, with its
   trigger and intended effect recorded. Versions give provenance and rollback. The immutable base prompt is
   never rewritten.
5. **Why it matters.** The paper frames the harness as a "membrane": a model should fail a task because the
   task is beyond it, not because the harness dropped state or limited its actions. Reported effects include
   ARC-AGI-3 going from 30% to 95.5% with Opus 5, and competitive long-context scores (OOLONG, LongBench)
   against native harnesses. The paper also reports the central risk of online refinement: in one Factorio
   trace the agent found a cheat and saved it as a skill, which is why auditable rollback matters.

## Concept to code map

| Paper concept | Where it lives here |
|---|---|
| Persistent IPython REPL per session | `rlm/kernel.py` `Kernel` (jupyter_client `KernelManager`, cwd = project dir, per-cell timeout with interrupt) |
| Long context kept out of the prompt, bound as a variable | `rlm/repl_api.py` `install()` loads the file into `ctx` inside the kernel; `rlm/agent.py` sends only `len(ctx)` and 3 short lines |
| Model writes cells; only truncated output returns (L2 to L1) | `rlm/agent.py` `run_agent()` loop, `truncate()` (1000 head + 500 tail chars) |
| Programmatic inspection helpers | `rlm/repl_api.py` `grep()`, `chunk()` (plus plain Python: slicing, `re`, `len`) |
| `llm()` cheap sub-call | `rlm/repl_api.py` `install().llm` (role `sub` in the backend) |
| `rlm()` recursive child with its own kernel and log | `rlm/repl_api.py` `install().rlm` calls `agent.run_agent(depth+1)`; child log `runs/<child>.jsonl`, child ctx `runs/<child>.ctx.txt` |
| Background child (`rlm.spawn`) | `rlm/repl_api.py` `rlm.spawn` returns a `concurrent.futures.Future` |
| Depth limit | `RLM_MAX_DEPTH` (default 2); at the limit `rlm()` degrades to `llm()` |
| Append-only event history | `rlm/trace.py` `SessionLog` (one JSON line per event: `session_start`, `model_call`, `cell`, `subcall`, `final`) |
| Resource accounting over root + descendants | `rlm/trace.py` `summarize()` (child totals reported up through `subcall` events) |
| Agents View (inspect the session tree) | `rlm trace <id>`: `rlm/trace.py` `render_tree()` prints each cell's code and nested sub-calls with per-call tokens |
| Continual Harness CRUD (prompts, skills, memory) | `rlm/harness.py` `Harness.create/update/delete/read`, files `harness_state/{prompts,skills,memory}.json`; exposed in the kernel as `harness` / `rlm.harness` |
| Supplemental state assembled into the next invocation | `Harness.system_prompt()`; skills are `exec`'d into the kernel at boot |
| Immutable base prompt | `harness_state/base_system_prompt.md` (copied from `rlm/base_system_prompt.md`); no write path exists, `validate()` rejects any other target |
| Versioned edits with trigger, diff, outcome | `Harness.apply_edit()` writes `harness_state/snapshots/sNNNN/{meta.json, <file>.json.before}` |
| `/refine`: smallest evidence-backed edit from the last trajectory | `rlm/refine.py` `refine()`: one JSON edit, evidence must be a verbatim quote from the trajectory, size cap, retry on rejection |
| `/rollback <id>` | `Harness.rollback()` copies the pre-edit bytes back |
| Score at fixed expenditure | `rlm/bench.py` reports accuracy next to total tokens and recursion depth for each arm |

## What I simplified, and why

- **Synchronous `rlm()` instead of daemon-backed async sessions with message queues.** A blocking call that
  returns the child's answer shows the recursion and accounting without a daemon. `rlm.spawn` covers the
  background case with a thread pool; there is no agent-to-agent messaging, no parent/child follow-ups and
  no recovery after restart.
- **Stateless model calls.** Each root turn re-sends the (small) conversation to `claude -p` as one
  flattened prompt with tools disabled. The paper's sessions keep live context; here the kernel is the only
  state that persists between turns, which is the part the mechanism needs.
- **No compaction.** Instead, a hard budget: before each root call `enforce_cap()` measures how many prompt
  characters are verbatim corpus text (64-char shingles) and elides the oldest kernel outputs until it is at
  most 8000. `rlm check-logs` re-measures every logged prompt independently.
- **Three harness types, not four.** Subagent specs are left out; `rlm()` takes a plain prompt.
- **`/refine` is one model call over one trajectory**, not a background process over many events, and an
  edit applies immediately rather than at a turn boundary of a running session.
- **Rollback restores only the file the edit touched**, to its exact pre-edit bytes. Rolling back an older
  edit also discards later edits to that same file; that is stated in the README.
- **Benchmark leak guard.** In `rlm bench`, refine edits may not contain any gold answer of the bench
  questions, because the refine rounds learn from the same questions that arm (c) is scored on.
- **No sandbox.** The kernel runs as the user in the project directory, as the paper's own harness does by
  default ("permissions of the runtime environment").
