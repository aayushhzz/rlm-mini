"""The RLM loop: the root model writes cells, the kernel runs them, truncated output goes back."""
import os
import re
from pathlib import Path

from . import PROJECT_DIR, home_dir
from .harness import Harness
from .kernel import Kernel
from .repl_api import FINAL_CLOSE, FINAL_OPEN
from .trace import SessionLog, corpus_chars, corpus_index, new_id, prompt_text, summarize

OUT_HEAD, OUT_TAIL = 1000, 500  # cell output returned to the model: head + tail chars
CORPUS_CAP = int(os.environ.get("RLM_CORPUS_CAP", 8000))  # max corpus chars in any root prompt
MAX_DEPTH = int(os.environ.get("RLM_MAX_DEPTH", 2))
ELIDED = "[older output elided to keep corpus text in the prompt under the cap]"

_CODE = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.S)
_FINAL_TEXT = re.compile(r"^\s*FINAL:\s*(.+)$", re.M)
_FINAL_MARK = re.compile(re.escape(FINAL_OPEN) + r"(.*?)" + re.escape(FINAL_CLOSE), re.S)


def truncate(text):
    if len(text) <= OUT_HEAD + OUT_TAIL:
        return text
    return "%s\n...[%d chars cut]...\n%s" % (text[:OUT_HEAD], len(text) - OUT_HEAD - OUT_TAIL, text[-OUT_TAIL:])


def enforce_cap(system, messages, index, cap):
    """Elide the oldest kernel outputs until the prompt holds at most `cap` corpus chars."""
    msgs = [dict(m) for m in messages]
    n = corpus_chars(prompt_text(system, msgs), index)
    for m in msgs:
        if n <= cap:
            break
        if m["role"] == "user" and m.get("output"):
            m["content"] = ELIDED
            n = corpus_chars(prompt_text(system, msgs), index)
    return [{"role": m["role"], "content": m["content"]} for m in msgs], n


def run_agent(question, ctx_path=None, ctx_text=None, *, backend, home=None, runs_dir=None, depth=0,
              parent=None, session_id=None, max_turns=8, cap=CORPUS_CAP, max_depth=MAX_DEPTH):
    home = home_dir(home)
    runs_dir = Path(runs_dir or home / "runs")
    sid = session_id or new_id()
    log_path = runs_dir / ("%s.jsonl" % sid)
    if ctx_text is not None:  # a child's sub_ctx is written to disk and loaded by its own kernel
        ctx_path = runs_dir / ("%s.ctx.txt" % sid)
        runs_dir.mkdir(parents=True, exist_ok=True)
        ctx_path.write_text(ctx_text)
    ctx_path = Path(ctx_path).resolve()
    ctx = ctx_path.read_text()
    index = corpus_index(ctx)
    log = SessionLog(log_path)
    harness = Harness(home)
    system = harness.system_prompt()
    log.event("session_start", session=sid, parent=parent, depth=depth, question=question,
              ctx_path=str(ctx_path), ctx_chars=len(ctx), backend=backend.name)

    kernel = Kernel(cwd=home, env={"RLM_BACKEND": backend.name})
    answer = None
    try:
        cfg = {"ctx_path": str(ctx_path), "log_path": str(log_path), "home": str(home),
               "session": sid, "depth": depth, "max_depth": max_depth}
        boot = kernel.execute("import sys; sys.path.insert(0, %r)\nfrom rlm.repl_api import install as _i\n"
                              "_i(globals(), %r); del _i" % (str(PROJECT_DIR), cfg))
        if "Error" in boot:
            raise RuntimeError("kernel bootstrap failed:\n" + boot)
        first_lines = "\n".join([ln[:120] for ln in ctx[:400].splitlines() if ln.strip()][:3])
        messages = [{"role": "user", "content":
                     "Question: %s\n\n`ctx` is a str of %d characters. Its first lines look like:\n%s\n\n"
                     "Write your first cell." % (question, len(ctx), first_lines)}]
        for turn in range(1, max_turns + 2):
            last = turn > max_turns
            if last:
                messages.append({"role": "user", "content": "Turn limit reached. Reply with one line: FINAL: <answer>"})
            msgs, n_corpus = enforce_cap(system, messages, index, cap)
            text, usage = backend.complete(system, msgs, role="root")
            log.event("model_call", turn=turn, system=system, messages=msgs, response=text,
                      tokens=usage, corpus_chars=n_corpus)
            messages.append({"role": "assistant", "content": text})
            code = _CODE.search(text)
            if not code or last:
                m = _FINAL_TEXT.search(text)
                if m or last:
                    answer = (m.group(1) if m else text).strip()
                    break
                messages.append({"role": "user", "content": "Reply with one ```python cell, or FINAL: <answer>."})
                continue
            out = kernel.execute(code.group(1))
            log.event("cell", turn=turn, code=code.group(1), output=out[:20000])
            fin = _FINAL_MARK.search(out)
            if fin:
                answer = fin.group(1).strip()
                break
            messages.append({"role": "user", "content": "Output:\n" + (truncate(out) or "(no output)"),
                             "output": True})
    finally:
        kernel.shutdown()
    log.event("final", answer=answer)
    res = summarize(log_path)
    res["log_path"] = str(log_path)
    return res
