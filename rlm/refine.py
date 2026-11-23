"""/refine: review one trajectory and make the single smallest evidence-backed harness edit."""
import json
import re
from pathlib import Path

from .harness import FILES, Harness
from .trace import load

REFINE_SYSTEM = """You improve the supplemental harness of a code-writing agent that answers questions about
a long text `ctx` from a Python kernel (helpers: grep, chunk, llm, rlm, FINAL). You review one trajectory
and make exactly ONE smallest edit that would most help future, different questions.

Allowed targets: "prompts" (a behavioural rule), "memory" (a durable fact about the data's format or the
tools), "skills" (a small reusable Python function the kernel will define; value is
{"description": str, "code": str}). You cannot edit the base system prompt.
Rules: one edit only; keep it short and general; never store the answer to this specific question;
"evidence" must be a short verbatim quote (under 300 chars) copied from the trajectory that motivates it.

Reply with only a JSON object:
{"file": "prompts|memory|skills", "op": "create|update|delete", "key": "...", "value": ..., "evidence": "...", "rationale": "..."}"""


def _norm(s):
    return " ".join(str(s).split())


def trajectory_text(log_path, verdict=None):
    ev = load(log_path)
    start = next(e for e in ev if e["kind"] == "session_start")
    out = ["QUESTION: %s" % start["question"], "ctx: %d chars" % start["ctx_chars"]]
    for e in ev:
        if e["kind"] == "cell":
            out.append("--- turn %d code:\n%s\n--- output:\n%s" % (e["turn"], e["code"].strip(), e["output"][:600]))
        elif e["kind"] == "subcall":
            out.append("--- %s() sub-call: %s -> %s" % (e["sub"], e["prompt"][:150], e["response"][:150]))
        elif e["kind"] == "final":
            out.append("FINAL ANSWER: %s" % e["answer"])
    if verdict:
        out.append("VERDICT: %s" % verdict)
    return "\n".join(out)


def latest_root_session(home):
    logs = sorted(Path(home).glob("runs/*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    for p in logs:
        start = next((e for e in load(p) if e["kind"] == "session_start"), None)
        if start and start["depth"] == 0:
            return p
    raise FileNotFoundError("no root trajectory in %s/runs; run `rlm ask` first" % home)


def parse_edit(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("no JSON object in the reply")
    return json.loads(m.group(0))


def check_edit(edit, harness, traj, forbidden=()):
    if not isinstance(edit, dict):
        raise ValueError("reply must be one JSON object (one edit)")
    if edit.get("file") not in FILES:
        raise ValueError("file must be one of %s; the base system prompt is not editable" % (FILES,))
    harness.validate(edit)
    ev = _norm(edit.get("evidence", ""))
    if not ev or len(ev) > 400 or ev not in _norm(traj):
        raise ValueError("evidence must be a short verbatim quote from the trajectory")
    blob = json.dumps(edit.get("value", "")).lower()
    for gold in forbidden:  # bench guard: a refine edit may not memorise a gold answer
        if len(str(gold)) >= 2 and re.search(r"(?<![\w.-])%s(?![\w-])" % re.escape(str(gold).lower()), blob):
            raise ValueError("edit stores a task-specific answer (%r); keep it general" % gold)


def refine(home, backend, log_path=None, verdict=None, forbidden=(), trigger="manual /refine", retries=2):
    """Ask the model for one edit, validate it, apply it with a snapshot. Returns (snapshot_id, edit, usage)."""
    harness = Harness(home)
    log_path = Path(log_path) if log_path else latest_root_session(home)
    traj = trajectory_text(log_path, verdict)
    state = {f: harness.read(f) for f in FILES}
    prompt = "TRAJECTORY:\n%s\n\nCURRENT HARNESS STATE:\n%s" % (traj, json.dumps(state, indent=1))
    usage = {"input": 0, "output": 0}
    for _ in range(retries + 1):
        text, u = backend.complete(REFINE_SYSTEM, [{"role": "user", "content": prompt}], role="root")
        usage["input"] += u["input"]
        usage["output"] += u["output"]
        try:
            edit = parse_edit(text)
            check_edit(edit, harness, traj, forbidden)
            break
        except ValueError as e:
            prompt += "\n\nYour previous reply was rejected: %s. Reply with one corrected JSON edit." % e
    else:
        raise ValueError("no valid edit after %d attempts" % (retries + 1))
    core = {k: edit[k] for k in ("file", "op", "key", "value") if k in edit}
    sid = harness.apply_edit(core, trigger=trigger, evidence=edit["evidence"],
                             rationale=edit.get("rationale", ""), session=log_path.stem)
    return sid, edit, usage
