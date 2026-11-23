# rlm-mini

A small, working Recursive Language Model (RLM) harness with a Continual Harness, built from the paper
**Prime Agent: A Self-Improving RLM Harness** (Karten, Zhang, et al., arXiv
[2608.23552](https://arxiv.org/abs/2608.23552), official code at
[PrimeIntellect-ai/prime-agent](https://github.com/PrimeIntellect-ai/prime-agent)). This is an independent
re-implementation of the core idea, not a port of the official code. `PAPER_NOTES.md` maps each paper
concept to the file that implements it and lists what was simplified.

## What the paper is, and why it matters

A model can only reason over what is in its context window. Prime Agent gives the model one tool, a
persistent Python kernel, and puts the long input in that kernel as a variable instead of in the prompt.
The model writes code to inspect it (slicing, regex, loops, `len`), and it can call models from inside
that code: `llm()` for a cheap sub-call and `rlm()` for a recursive child agent with its own kernel. Only
short cell outputs return to the root model, so a 300k-character input costs a few thousand prompt tokens
per turn. On top of this, the Continual Harness keeps prompts, memories and skills on disk. `/refine`
turns evidence from a past trajectory into one small, versioned edit, and `/rollback` undoes it. The base
system prompt is never touched. The paper reports large gains from this design, for example ARC-AGI-3
going from 30% to 95.5%, and argues that harness failures should not be mistaken for model failures.

## What is here

| Command | What it does |
|---|---|
| `rlm ask "question"` | Root model answers over the ~295k-char corpus bound as `ctx` (or `--ctx FILE`). The corpus never goes into the prompt. |
| `rlm trace <session>` | Prints the code the model wrote, cell by cell, with `llm()`/`rlm()` sub-calls nested as a tree and per-call tokens. |
| `rlm check-logs` | Re-measures every logged root and child prompt and fails if any holds more than 8000 chars of corpus text. |
| `rlm refine [--session id]` | Reviews the last trajectory and writes exactly one edit to `harness_state/` with a snapshot id, a verbatim evidence quote, a rationale and a diff. |
| `rlm rollback <id>` | Restores the file that edit touched to its exact pre-edit bytes. |
| `rlm snapshots` | Lists snapshots with trigger and outcome. |
| `rlm bench [--limit N] [--ids q01,..] [--rounds 3]` | Three arms on 20 needle and aggregation questions: (a) prompt stuffing, (b) RLM, (c) RLM after 3 `/refine` rounds. Writes accuracy, total tokens, recursion depth and sub-calls to `results.md`. |

Kernel helpers: `ctx`, `grep(text, pattern)`, `chunk(text, n)`, `llm(prompt)`, `rlm(prompt, sub_ctx)`,
`rlm.spawn(prompt, sub_ctx)` (returns a Future), `harness` (also `rlm.harness`: CRUD over
`harness_state/{prompts,skills,memory}.json`), and `FINAL(answer)`.

The corpus (`rlm/corpus.py`) is generated deterministically: a synthetic repo dump (service configs,
handlers, padding modules) interleaved with four days of logs, about 400 free-text operator notes and
planted facts. All 20 gold answers are computed from the generator, so they are exact. Several needles
sit past the first 80k characters, which the stuffing arm cannot see.

## Model backends

All model calls go through one interface in `rlm/backends.py`: `complete(system, messages, role)`.

- **`claude` (default).** Spawns `claude -p --output-format json` with built-in tools disabled
  (`--tools ""`), no session persistence, no settings, MCP or slash commands, run from an empty temp
  directory. It uses the Claude Code CLI's logged-in claude.ai subscription. No API key is used, and
  `ANTHROPIC_API_KEY` is removed from the subprocess environment so the login is always used. The model
  is the CLI's default (it reported `claude-haiku-5-5` early on and `claude-opus-5-5` later). Override it with `--model sonnet` or
  `--model claude-sonnet-5-5` (`RLM_MODEL`); `--sub-model` (`RLM_SUB_MODEL`) sets the `llm()` model.
- **`ollama` (optional, local).** `qwen3:8b` as root and `qwen3:4b-instruct` for `llm()`. It uses
  Ollama's native `/api/chat` instead of the OpenAI-compatible endpoint, because only the native API can
  set `num_ctx` per request (32768 for the stuffing arm), and it turns off qwen3's thinking.
- **`fake`.** Scripted and rule-based, for offline tests.

Choose with `--backend` or `RLM_BACKEND`. The kernel and every recursive child inherit the choice through
the environment.

**Terms note.** This uses your personal Claude subscription through the local Claude Code login, which is
meant for your own local use. If you share or deploy it for other people, switch to API-key
authentication as Anthropic's Agent SDK terms require.

## Setup

```bash
cd rlm-mini
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
claude auth status          # needs loggedIn: true (claude.ai)
# optional local backend:
ollama pull qwen3:8b && ollama pull qwen3:4b-instruct
```

## Run

```bash
.venv/bin/rlm ask "In the free-text operator notes, one vendor is the target of a complaint. Which vendor?"
.venv/bin/rlm trace <session-id-printed-above>
.venv/bin/rlm check-logs
.venv/bin/rlm refine && .venv/bin/rlm snapshots
.venv/bin/rlm rollback s0001
.venv/bin/rlm bench --ids q01,q08,q10,q19,q20      # small Claude run
.venv/bin/rlm --backend ollama bench               # all 20 questions, local
.venv/bin/python -m pytest -q                      # offline tests, fake backend
```

Settings: `RLM_CELL_TIMEOUT` (240 s per cell), `RLM_CORPUS_CAP` (8000), `RLM_MAX_DEPTH` (2),
`RLM_EFFORT` (`--effort` passed to `claude -p`, default `low`), `RLM_HOME` (state directory; default is this
folder).

State on disk: `runs/<session>.jsonl` holds the append-only event log per session, and a child's
`sub_ctx` is saved as `runs/<child>.ctx.txt`. `harness_state/` holds the Continual Harness and its
`snapshots/`. `bench/<bench-id>/` gets its own fresh `harness_state/` and `runs/`, so a bench never
changes your main harness.

## Demo output

All runs below were real calls through the local Claude Code CLI (claude.ai Pro login, no API key) on
2026-11-20. The CLI's default model was reported as `claude-haiku-5-5` for some early calls and
`claude-opus-5-5` for later ones (the default is the CLI's choice). Raw output is in `demo/`, and the
logs are in `runs/` and `bench/`.

**`rlm ask`** (`demo/ask.txt`, `demo/trace_ask.txt`): correct after 9 turns, 35,092 tokens. The model
found the notes with `grep`, removed the boilerplate and printed only the rare lines.

```
$ rlm ask "In the free-text operator notes, one vendor is the target of a complaint. Which vendor?"
session=53654f7349 tokens=35092 (in 32419 / out 2673) depth_used=0 subcalls=0
Zentrix Cloud
```

**Recursion, `rlm trace`** (`demo/trace_recursive.txt`). The question asked the model to delegate. The
root passed 402 note lines (48,726 chars) to an `rlm()` child. The child mapped `llm()` over
15,000-char chunks, which is depth 2 with 5 sub-calls in total.

```
session 9f73c2f87a (depth 0) Q: Collect the [ops-note] lines into one string and delegate to a chil...
  answer='Zentrix Cloud'  tokens: in=38474 out=967 total=39441  depth_used=2
├─ turn 1  root call [in=1703 out=92] claude-opus-5-5
│    > lines = grep(ctx, r"\[ops-note\]")
│    > s = "\n".join(lines); print(len(lines), len(s))
│    > r = rlm("Which of these operator notes complains about a vendor? Reply with just the vendor name.", s); print(r)
│    └─ rlm() [in=32866 out=754] prompt='Which of these operator notes complains about a...' -> 'Zentrix Cloud'
│         session 367ff55750 (depth 1) Q: Which of these operator notes complains about a vendor? ...
│         ├─ turn 2  root call [in=2000 out=138] claude-opus-5-5
│         │    > res=[llm("These are operator notes. Does any note complain about a vendor ...\n\n"+p) for p in chunk(ctx,15000)]
│         │    └─ llm() [in=7967 out=117] prompt='These are operator notes. Does any note complai...' -> 'NONE. Every vendor mention (Kestrel D...'
│         │    └─ llm() [in=7921 out=89] ...
│         │    └─ llm() [in=7919 out=97] ... -> 'Yes. Vendor: **Zentrix Cloud** > 2026...'
│         │    └─ llm() [in=2708 out=89] ...
│         └─ FINAL 'Zentrix Cloud'
...
└─ FINAL 'Zentrix Cloud'
```

**Corpus cap, `rlm check-logs`** (`demo/check_logs.txt`), run over every logged prompt from all the runs
above, children included:

```
checked 85 logged root/child prompts; max corpus chars in one prompt = 3432 (cap 8000)
PASS
```

**`rlm refine` and `rlm rollback`** (`demo/refine.txt`, `demo/rollback.txt`). The refine ran on the
`ask` trajectory and wrote one memory entry. Its evidence was the verbatim log line the agent had found.
Rollback then restored every file: the sha256 of every `harness_state` file matched the hashes taken
before the refine (`demo/sha_before.txt` vs `demo/sha_after.txt`).

```
snapshot s0001: create memory/operator_notes_format
evidence: 2026-03-03T20:42:23Z [ops-note] NOTE author=mira: Honestly fed up with Zentrix Cloud: ...
rationale: The agent wasted several turns on file-header and keyword searches ...
$ rlm rollback s0001
restored harness_state/memory.json to its bytes before s0001 (sha256 ca3d163bab05)
byte-identical after rollback
```

**`rlm bench`, a small Claude run on 5 of the 20 questions** (`results.md`, `demo/bench_claude.log`).
To save Pro usage, it covered two aggregation questions, one late needle and both free-text questions:

| Arm | Accuracy | Total tokens | Mean tokens / question | Max recursion depth | Sub-calls |
|---|---|---|---|---|---|
| (a) prompt stuffing, first 80k chars | 0/5 (0%) | 202,112 | 40,422 | 0 | 0 |
| (b) RLM | 5/5 (100%) | 88,093 | 17,618 | 0 | 0 |
| (c) RLM + 3 refines | 5/5 (100%) | 86,248 | 17,249 | 0 | 0 |

The three refine edits were memory notes on the log-line format, on the `status=` field (the model had
matched `latency_ms=503` by mistake) and on the `===== FILE:` block format. Each cites a verbatim
evidence quote. Stuffing saw only the first 80k chars. It miscounted both aggregations (13 vs 64,
6 vs 17) and could not see the needles.

A second 2-question run (`results_claude_semantic.md`) came after one sentence was added to the base
prompt nudging `llm()` for free-text questions. It went 0/2, 2/2, 2/2. In arm (c), one root turn on
q19 reported 128k output tokens for a 106-character reply, probably runaway thinking, so that arm's
total is 261,923 tokens. Since then the backend passes `--effort low` by default (`RLM_EFFORT`).

In these bench runs the models answered with `grep` and plain Python alone (depth 0). The recursive
sub-call trajectory shown above is from `rlm ask`. The offline bench test also exercises sub-calls.

**Not run: the full 20-question bench.** I started it on Ollama twice. With `num_ctx` at 32768,
`qwen3:8b` takes 9.9 GB, and on this 16 GB machine the system went 10 GB into swap; the first
stuffing call never returned, so I stopped it. A standalone stuffing call worked once and took 96 s,
but Ollama only evaluated 16,386 prompt tokens, which means it silently cut the start of the 80k
chars. A single Ollama `ask` (`demo/ask_ollama.txt`) ran in 13 s, but `qwen3:8b` called `FINAL` in its
first cell and gave a wrong answer. `rlm bench` with no flags runs all 20 questions. On Claude that is
roughly 1.5M tokens.

## Tests

`.venv/bin/python -m pytest -q` runs 14 tests offline, using a real Jupyter kernel and the fake backend.
No Claude or Ollama calls are made. The tests cover:

- `ctx` is bound in the kernel; prompts stay under 8k chars while the kernel computes the exact count.
- A greedy model that prints 5k-char slices of `ctx` every turn still stays under the cap (old outputs
  are elided), checked by `check_logs` over the written logs.
- `llm()` and `rlm()` sub-calls run (the child gets its own kernel and log) and render as a tree with
  per-call tokens.
- The per-cell timeout interrupts the cell and the kernel survives.
- `refine` writes exactly one edit (one file, one key, one snapshot) with evidence, and `rollback`
  restores the bytes exactly, including create, update and delete chains.
- The base system prompt's sha256 is unchanged after refines, including one where the model tries to
  edit it. Fabricated evidence, leaked gold answers and skills that do not compile are rejected.
- Gold answers match the generated text, scoring is whole-token, and an offline 3-arm bench writes
  `results.md`.

## Limits

- **Not a sandbox.** Model-written code runs as you, with your permissions, in this folder, and can read
  and write files and use the network. Use it only on your own machine with inputs you trust.
- Each root turn is a fresh, stateless `claude -p` call that re-sends the short conversation. The CLI
  adds some fixed overhead per call (about 1k tokens of its own system context, mostly cached).
- `rlm()` blocks; `rlm.spawn` uses threads. There is no daemon, no agent-to-agent messaging, no
  compaction and no restart recovery (see `PAPER_NOTES.md`).
- Rolling back an older snapshot restores that file as it was before that edit, so later edits to the
  same file are discarded.
- The stuffing arm sends the first 80,000 corpus characters. Claude counts them as about 40k tokens.
  Ollama, at `num_ctx` 32768, cut the prompt to about 16k tokens.
- The bench's refine rounds learn from the same questions arm (c) is scored on. Edits that contain a gold
  answer are rejected, but this is still not a held-out evaluation. With so few questions, small
  accuracy differences are noise.
- Recursion is the model's choice. Strong models often answer with `grep` alone, so `llm()`/`rlm()`
  sub-calls show up mostly on the free-text questions.
