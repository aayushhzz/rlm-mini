"""rlm bench: (a) prompt stuffing truncated to a 32k-token window, (b) RLM, (c) RLM after N /refine rounds."""
import re
import sys
import time
from pathlib import Path

from .agent import run_agent
from .corpus import ensure_corpus
from .harness import Harness
from .refine import refine
from .trace import SessionLog, new_id, summarize

STUFF_NUM_CTX = 32768
STUFF_CHARS = 80000  # ~32k tokens of this log-heavy text, leaving room for the question and answer
STUFF_SYSTEM = "Answer the question using only the document. Reply with just the answer value, no other words."


def score(answer, gold):
    """Gold must appear in the answer as a whole token (so 4 does not match 14)."""
    a = " ".join(str(answer or "").lower().replace("`", "").replace("*", "").split())
    return re.search(r"(?<![\w.-])%s(?![\w-])" % re.escape(gold.lower()), a) is not None


def stuff_one(q, corpus_path, backend, runs_dir):
    sid = new_id()
    log = SessionLog(Path(runs_dir) / ("%s.jsonl" % sid))
    text = Path(corpus_path).read_text()
    log.event("session_start", session=sid, parent=None, depth=0, question=q["question"],
              ctx_path=str(corpus_path), ctx_chars=len(text), backend=backend.name)
    # Corpus first, question last, so repeated calls share a cacheable prefix.
    msgs = [{"role": "user", "content": "DOCUMENT (truncated):\n%s\n\nQUESTION: %s" % (text[:STUFF_CHARS], q["question"])}]
    answer, usage = backend.complete(STUFF_SYSTEM, msgs, role="root", num_ctx=STUFF_NUM_CTX)
    # The full prompt is not stored again: it is the first STUFF_CHARS of the corpus plus the question.
    log.event("model_call", turn=1, arm="stuffing", system=STUFF_SYSTEM,
              messages=[{"role": "user", "content": "<corpus[:%d]> QUESTION: %s" % (STUFF_CHARS, q["question"])}],
              response=answer, tokens=usage, corpus_chars=min(STUFF_CHARS, len(text)))
    log.event("final", answer=answer.strip())
    return summarize(log.path)


def _safe(fn, *a, **kw):
    try:
        return fn(*a, **kw)
    except Exception as e:  # one failed question must not lose the whole bench
        print("  ! %s" % e, file=sys.stderr)
        return {"session": None, "answer": "ERROR: %s" % str(e)[:80], "tokens": 0, "depth_used": 0, "subcalls": 0}


def _arm_rlm(qs, corpus_path, backend, bhome, label):
    out = {}
    for q in qs:
        r = _safe(run_agent, q["question"], ctx_path=corpus_path, backend=backend, home=bhome)
        r["ok"] = score(r["answer"], q["gold"])
        print("  [%s] %s %s answer=%r gold=%r tokens=%d depth=%d" % (label, q["id"], "ok " if r["ok"] else "BAD",
              str(r["answer"])[:50], q["gold"], r["tokens"], r["depth_used"]), file=sys.stderr)
        out[q["id"]] = r
    return out


def run_bench(home, backend, limit=None, rounds=3, out_path=None, ids=None):
    home = Path(home)
    corpus_path, qs = ensure_corpus(home)
    qs = [q for q in qs if not ids or q["id"] in ids]
    qs = qs[:limit] if limit else qs
    bench_id = time.strftime("%Y%m%d-%H%M%S")
    bhome = home / "bench" / bench_id  # fresh harness_state and runs/ for this bench
    harness = Harness(bhome)
    golds = [q["gold"] for q in qs]
    t0 = time.time()

    print("arm (a) stuffing, %d questions" % len(qs), file=sys.stderr)
    arm_a = {}
    for q in qs:
        r = _safe(stuff_one, q, corpus_path, backend, bhome / "runs")
        r["ok"] = score(r["answer"], q["gold"])
        print("  [stuff] %s %s answer=%r gold=%r tokens=%d" % (q["id"], "ok " if r["ok"] else "BAD", str(r["answer"])[:50], q["gold"], r["tokens"]), file=sys.stderr)
        arm_a[q["id"]] = r

    print("arm (b) RLM", file=sys.stderr)
    arm_b = _arm_rlm(qs, corpus_path, backend, bhome, "rlm")

    print("%d /refine rounds" % rounds, file=sys.stderr)
    order = [q for q in qs if not arm_b[q["id"]]["ok"]] + [q for q in qs if arm_b[q["id"]]["ok"]]
    order = [q for q in order if arm_b[q["id"]]["session"]]
    edits, refine_tokens = [], 0
    for r in range(rounds):
        if not order:
            break
        q = order[r % len(order)]
        res = arm_b[q["id"]]
        verdict = "correct" if res["ok"] else "incorrect (the expected answer is withheld)"
        log = bhome / "runs" / ("%s.jsonl" % res["session"])
        try:
            sid, edit, usage = refine(bhome, backend, log_path=log, verdict=verdict, forbidden=golds,
                                      trigger="bench round %d on %s (%s)" % (r + 1, q["id"], verdict.split()[0]))
        except Exception as e:
            print("  ! refine round %d failed: %s" % (r + 1, e), file=sys.stderr)
            continue
        refine_tokens += usage["input"] + usage["output"]
        edits.append((sid, q["id"], edit))
        print("  refine %s: %s %s/%s" % (sid, edit["op"], edit["file"], edit["key"]), file=sys.stderr)

    print("arm (c) RLM + %d refines" % len(edits), file=sys.stderr)
    arm_c = _arm_rlm(qs, corpus_path, backend, bhome, "rlm+refine")
    acc_c = sum(r["ok"] for r in arm_c.values()) / len(qs)
    for sid, _, _ in edits:
        harness.set_outcome(sid, {"arm_c_accuracy": acc_c, "bench": bench_id})

    md = _report(qs, arm_a, arm_b, arm_c, edits, refine_tokens, backend, bench_id, time.time() - t0,
                 len(corpus_path.read_text()))
    out_path = Path(out_path or home / "results.md")
    out_path.write_text(md)
    (bhome / "results.md").write_text(md)
    return md


def _report(qs, a, b, c, edits, refine_tokens, backend, bench_id, secs, corpus_chars):
    n = len(qs)
    lines = ["# rlm bench results", "",
             "- bench id: `%s` (logs in `bench/%s/runs/`), backend: `%s`, questions: %d, wall time: %.0fs" % (bench_id, bench_id, backend.name, n, secs),
             "- stuffing arm: first %d chars of the %s-char corpus (Ollama: num_ctx 32768; Claude counts it as ~40k tokens), one call per question" % (STUFF_CHARS, "{:,}".format(corpus_chars)),
             "- refine tokens (not in the arm totals): %d" % refine_tokens, "",
             "| Arm | Accuracy | Total tokens | Mean tokens / question | Max recursion depth | Sub-calls (llm/rlm) |",
             "|---|---|---|---|---|---|"]
    for name, arm in [("(a) prompt stuffing, first 80k chars", a), ("(b) RLM", b), ("(c) RLM + %d refines" % len(edits), c)]:
        ok = sum(r["ok"] for r in arm.values())
        tok = sum(r["tokens"] for r in arm.values())
        lines.append("| %s | %d/%d (%.0f%%) | %s | %s | %d | %d |" % (
            name, ok, n, 100.0 * ok / n, "{:,}".format(tok), "{:,}".format(tok // n),
            max(r["depth_used"] for r in arm.values()), sum(r["subcalls"] for r in arm.values())))
    lines += ["", "## Per question", "", "| id | type | gold | stuffing | RLM | RLM+refine | RLM session |", "|---|---|---|---|---|---|---|"]
    cell = lambda r: ("ok" if r["ok"] else "x") + " `%s`" % " ".join(str(r["answer"]).split())[:28].replace("|", "/")
    for q in qs:
        i = q["id"]
        lines.append("| %s | %s | `%s` | %s | %s | %s | `%s` |" % (i, q["type"], q["gold"], cell(a[i]), cell(b[i]), cell(c[i]), b[i]["session"]))
    lines += ["", "## Refine edits (Continual Harness snapshots)", ""]
    for sid, qid, e in edits:
        lines.append("- `%s` from %s: %s `%s/%s` = %s  \n  evidence: \"%s\"  \n  rationale: %s" % (
            sid, qid, e["op"], e["file"], e["key"], " ".join(str(e.get("value", "")).split())[:160],
            " ".join(e["evidence"].split())[:160], " ".join(str(e.get("rationale", "")).split())[:160]))
    if not edits:
        lines.append("(none)")
    return "\n".join(lines) + "\n"
