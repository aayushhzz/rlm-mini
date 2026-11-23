"""Append-only JSONL session logs, token totals, the trace tree, and the corpus-in-prompt check."""
import json
import time
import uuid
from pathlib import Path

WINDOW = 64  # a prompt char counts as corpus text if it lies in a 64-char window copied from ctx


def new_id():
    return uuid.uuid4().hex[:10]


class SessionLog:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def event(self, kind, **fields):
        rec = dict(t=round(time.time(), 3), kind=kind, **fields)
        with open(self.path, "a") as f:  # one write per line; append-only
            f.write(json.dumps(rec) + "\n")


def find_log(session_id, home):
    """Locate runs/<id>.jsonl anywhere under home (bench runs live in bench/<id>/runs/). Prefix ok."""
    hits = sorted(Path(home).glob("**/runs/%s*.jsonl" % session_id))
    if not hits:
        raise FileNotFoundError("no session log for %r under %s" % (session_id, home))
    return hits[0]


def load(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def summarize(path):
    """Tokens (own model calls + sub-calls, children included via their reported totals) and depth used."""
    ev = load(path)
    start = next(e for e in ev if e["kind"] == "session_start")
    depth = start["depth"]
    tin = tout = 0
    depth_used, n_sub = depth, 0
    for e in ev:
        if e["kind"] == "model_call":
            tin += e["tokens"]["input"]
            tout += e["tokens"]["output"]
        elif e["kind"] == "subcall":
            n_sub += 1
            tin += e["tokens"]["input"]
            tout += e["tokens"]["output"]
            if e["sub"] == "rlm":
                child = summarize(Path(path).parent / ("%s.jsonl" % e["child"]))
                depth_used = max(depth_used, child["depth_used"])
                n_sub += child["subcalls"]
            else:
                depth_used = max(depth_used, depth + 1)
    final = next((e for e in ev if e["kind"] == "final"), {})
    return {"session": start["session"], "question": start["question"], "answer": final.get("answer"),
            "tokens_in": tin, "tokens_out": tout, "tokens": tin + tout,
            "depth_used": depth_used, "subcalls": n_sub}


def _short(s, n=90):
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[: n - 3] + "..."


def render_tree(path, prefix=""):
    """Cell-by-cell code the model wrote, with llm()/rlm() sub-calls nested as a tree."""
    ev = load(path)
    s = summarize(path)
    lines = ["%ssession %s (depth %d) Q: %s" % (prefix, s["session"], next(e for e in ev if e["kind"] == "session_start")["depth"], _short(s["question"], 70)),
             "%s  answer=%r  tokens: in=%d out=%d total=%d  depth_used=%d" % (prefix, _short(s["answer"], 60), s["tokens_in"], s["tokens_out"], s["tokens"], s["depth_used"])]
    pending = []  # sub-calls are logged while their cell runs; print them under the cell's code
    for e in ev:
        if e["kind"] == "model_call":
            lines.append("%s├─ turn %d  root call [in=%d out=%d] %s" % (prefix, e["turn"], e["tokens"]["input"], e["tokens"]["output"], e["tokens"].get("model", "")))
        elif e["kind"] == "cell":
            for cl in e["code"].rstrip().splitlines()[:25]:
                lines.append("%s│    > %s" % (prefix, cl))
            lines += pending
            pending = []
            lines.append("%s│    = %s" % (prefix, _short(e["output"], 100) or "(no output)"))
        elif e["kind"] == "subcall":
            pending.append("%s│    └─ %s() [in=%d out=%d] prompt=%r -> %r" % (prefix, e["sub"], e["tokens"]["input"], e["tokens"]["output"], _short(e["prompt"], 50), _short(e["response"], 40)))
            if e["sub"] == "rlm":
                pending += render_tree(Path(path).parent / ("%s.jsonl" % e["child"]), prefix + "│         ")
        elif e["kind"] == "final":
            lines.append("%s└─ FINAL %r" % (prefix, _short(e["answer"], 80)))
    return lines


def corpus_index(text):
    return {hash(text[i:i + WINDOW]) for i in range(max(0, len(text) - WINDOW + 1))}


def corpus_chars(prompt, index):
    """Number of prompt chars covered by 64-char windows that also occur verbatim in the corpus."""
    covered, end = 0, 0
    for i in range(len(prompt) - WINDOW + 1):
        if hash(prompt[i:i + WINDOW]) in index:
            start = max(i, end)
            covered += i + WINDOW - start
            end = i + WINDOW
    return covered


def prompt_text(system, messages):
    return system + "\n" + "\n".join(m["content"] for m in messages)


def check_logs(home, cap):
    """Re-measure every logged model prompt against its session's ctx. Returns (rows, max_chars)."""
    rows, indexes = [], {}
    for path in sorted(Path(home).glob("**/runs/*.jsonl")):
        ev = load(path)
        start = next((e for e in ev if e["kind"] == "session_start"), None)
        if not start or not start.get("ctx_path"):
            continue
        cp = start["ctx_path"]
        if cp not in indexes:
            indexes[cp] = corpus_index(Path(cp).read_text())
        for e in ev:
            if e["kind"] == "model_call" and e.get("arm") != "stuffing":
                n = corpus_chars(prompt_text(e["system"], e["messages"]), indexes[cp])
                rows.append((start["session"], start["depth"], e["turn"], n))
    worst = max((r[3] for r in rows), default=0)
    return rows, worst
