# rlm-mini Architecture

This document outlines the runtime architecture of `rlm-mini`. For paper background and a concept-to-code mapping against Prime Agent (arXiv:2608.23552), see [`PAPER_NOTES.md`](../PAPER_NOTES.md).

## 1. Persistent Kernel & REPL API ([`kernel.py`](../rlm/kernel.py), [`repl_api.py`](../rlm/repl_api.py))

- Each root or child session spawns its own persistent Jupyter Python kernel (`jupyter_client` + `ipykernel`) with a per-cell timeout (`RLM_CELL_TIMEOUT`, default `240s`). On timeout, the cell is interrupted while preserving kernel state.
- At bootstrap, `repl_api.install()` binds the input file as `ctx` and injects helpers into kernel globals:
  - `grep(text, pattern)` and `chunk(text, n)` for inspecting long inputs in Python
  - `llm(prompt)` for single-turn sub-model calls
  - `rlm(prompt, sub_ctx)` and `rlm.spawn(prompt, sub_ctx)` for recursive child sessions (up to `RLM_MAX_DEPTH`, default `2`)
  - `harness` (`rlm.harness`) for CRUD over `harness_state/{prompts,skills,memory}.json`
  - `FINAL(answer)` to emit a sentinel-wrapped final answer

## 2. Corpus-Cap Enforcement ([`agent.py`](../rlm/agent.py), [`trace.py`](../rlm/trace.py))

- Cell outputs returned to the model are truncated to head (`1,000` chars) + tail (`500` chars).
- Before every model call, `enforce_cap()` counts how many corpus characters appear in the assembled prompt using `corpus_index(ctx)` and elides oldest kernel output messages until the prompt holds at most `RLM_CORPUS_CAP` (`8,000`) corpus characters.
- Every session writes an append-only JSONL log to `runs/<session>.jsonl` (and child contexts to `runs/<child>.ctx.txt`), which `rlm check-logs` independently re-verifies.

## 3. Continual Harness & Snapshots ([`harness.py`](../rlm/harness.py), [`refine.py`](../rlm/refine.py))

- `rlm/base_system_prompt.md` is immutable; dynamic state lives in `harness_state/{prompts,skills,memory}.json`.
- `rlm refine` inspects a session trajectory and produces **exactly one** validated edit (one file, one key) with a verbatim evidence quote from the log, a rationale, and a diff. Fabricated evidence, gold-answer leaks, invalid Python skills, or attempts to touch the base prompt are rejected.
- Every edit writes a snapshot under `harness_state/snapshots/sNNNN/` storing the exact pre-edit file bytes and SHA-256 hash so `rlm rollback <id>` is byte-exact.
- `rlm bench` creates an isolated `bench/<bench-id>/` directory with its own `harness_state/` and `runs/` so benchmarks never mutate your main harness.
