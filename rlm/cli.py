"""rlm ask | refine | rollback | snapshots | bench | trace | check-logs"""
import argparse
import json
import os
import sys

from . import home_dir


def main(argv=None):
    ap = argparse.ArgumentParser(prog="rlm", description="Minimal RLM harness with a Continual Harness.")
    ap.add_argument("--backend", help="claude (default) | ollama | fake; same as RLM_BACKEND")
    ap.add_argument("--model", help="root model override (RLM_MODEL), e.g. sonnet or claude-sonnet-5-5")
    ap.add_argument("--sub-model", help="llm() sub-call model override (RLM_SUB_MODEL)")
    ap.add_argument("--home", help="state dir (default: project dir, or RLM_HOME)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("ask", help="answer a question over the corpus (or --ctx FILE)")
    p.add_argument("question")
    p.add_argument("--ctx", help="text file to bind as ctx (default: generated corpus)")
    p.add_argument("--max-turns", type=int, default=8)
    p = sub.add_parser("refine", help="one minimal, evidence-backed harness edit from the last trajectory")
    p.add_argument("--session", help="session id to review (default: latest root session)")
    p = sub.add_parser("rollback", help="restore the file an edit touched to its pre-edit bytes")
    p.add_argument("snapshot_id")
    sub.add_parser("snapshots", help="list harness snapshots")
    p = sub.add_parser("bench", help="3-arm benchmark, writes results.md")
    p.add_argument("--limit", type=int, help="only the first N of the 20 questions")
    p.add_argument("--ids", help="comma-separated question ids, e.g. q01,q10,q19")
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--out", help="results file (default: <home>/results.md)")
    p = sub.add_parser("trace", help="print a session as a tree of cells and sub-calls")
    p.add_argument("session_id")
    p = sub.add_parser("check-logs", help="verify no logged root prompt holds more than --cap corpus chars")
    p.add_argument("--cap", type=int, default=int(os.environ.get("RLM_CORPUS_CAP", 8000)))
    a = ap.parse_args(argv)

    # Env, not arguments, so kernels and recursive children inherit the same choice.
    for flag, var in [(a.backend, "RLM_BACKEND"), (a.model, "RLM_MODEL"), (a.sub_model, "RLM_SUB_MODEL")]:
        if flag:
            os.environ[var] = flag
    home = home_dir(a.home)

    from .backends import get_backend
    from .harness import Harness
    from .trace import check_logs, find_log, render_tree

    if a.cmd == "ask":
        from .agent import run_agent
        from .corpus import ensure_corpus
        ctx_path = a.ctx or ensure_corpus(home)[0]
        r = run_agent(a.question, ctx_path=ctx_path, backend=get_backend(), home=home, max_turns=a.max_turns)
        print(r["answer"])
        print("session=%s tokens=%d (in %d / out %d) depth_used=%d subcalls=%d" % (
            r["session"], r["tokens"], r["tokens_in"], r["tokens_out"], r["depth_used"], r["subcalls"]), file=sys.stderr)
    elif a.cmd == "refine":
        from .refine import refine
        log = find_log(a.session, home) if a.session else None
        sid, edit, usage = refine(home, get_backend(), log_path=log)
        m = Harness(home).meta(sid)
        print("snapshot %s: %s %s/%s\nevidence: %s\nrationale: %s\n%s" % (
            sid, edit["op"], edit["file"], edit["key"], m["evidence"], m["rationale"], m["diff"]))
    elif a.cmd == "rollback":
        m = Harness(home).rollback(a.snapshot_id)
        print("restored harness_state/%s.json to its bytes before %s (sha256 %s)" % (m["file"], a.snapshot_id, m["sha256_before"][:12]))
    elif a.cmd == "snapshots":
        for m in Harness(home).snapshots():
            print("%s  %s  %s %s/%s  outcome=%s" % (m["id"], m["created"], m["edit"]["op"], m["file"], m["edit"]["key"], json.dumps(m["outcome"])))
    elif a.cmd == "bench":
        from .bench import run_bench
        print(run_bench(home, get_backend(), limit=a.limit, rounds=a.rounds, out_path=a.out,
                        ids=a.ids.split(",") if a.ids else None))
    elif a.cmd == "trace":
        print("\n".join(render_tree(find_log(a.session_id, home))))
    elif a.cmd == "check-logs":
        rows, worst = check_logs(home, a.cap)
        print("checked %d logged root/child prompts; max corpus chars in one prompt = %d (cap %d)" % (len(rows), worst, a.cap))
        if worst > a.cap:
            print("FAIL: cap exceeded", file=sys.stderr)
            return 1
        print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
