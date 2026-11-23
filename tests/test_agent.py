"""RLM loop with a real kernel and the offline fake backend (no Claude or Ollama calls)."""
import re

from rlm.agent import ELIDED, run_agent
from rlm.backends import FakeBackend
from rlm.corpus import ensure_corpus
from rlm.kernel import Kernel
from rlm.trace import check_logs, load, render_tree


def cell(code):
    return "Let me look.\n```python\n%s\n```" % code


def test_ctx_is_bound_in_kernel_not_in_prompt(tmp_path, monkeypatch):
    monkeypatch.setenv("RLM_BACKEND", "fake")
    corpus, qs = ensure_corpus(tmp_path)
    fb = FakeBackend([cell("n = len(grep(ctx, r'\\[auth\\] ERROR'))\nprint(n)"), cell("FINAL(n)")])
    r = run_agent(qs[0]["question"], ctx_path=corpus, backend=fb, home=tmp_path)
    assert r["answer"] == qs[0]["gold"]  # the kernel computed the exact count
    text = corpus.read_text()
    assert len(text) > 250000
    for call in fb.calls:  # the full prompt is a few KB, nowhere near the corpus
        assert sum(len(m["content"]) for m in call["messages"]) + len(call["system"]) < 8000
    assert "Output:\n%s" % qs[0]["gold"] in fb.calls[1]["messages"][-1]["content"]


def test_root_prompts_stay_under_corpus_cap(tmp_path, monkeypatch):
    monkeypatch.setenv("RLM_BACKEND", "fake")
    corpus, _ = ensure_corpus(tmp_path)
    # A greedy model dumps 5k chars of ctx every turn; outputs are truncated and old ones elided.
    script = [cell("print(ctx[%d:%d])" % (i * 5000, i * 5000 + 5000)) for i in range(8)] + ["FINAL: done"]
    fb = FakeBackend(script)
    run_agent("dump it", ctx_path=corpus, backend=fb, home=tmp_path, cap=4000)
    rows, worst = check_logs(tmp_path, cap=4000)
    assert len(rows) == 9 and 0 < worst <= 4000
    assert any(ELIDED in m["content"] for m in fb.calls[-1]["messages"])


def test_recursive_llm_and_rlm_subcalls_render_as_tree(tmp_path, monkeypatch):
    monkeypatch.setenv("RLM_BACKEND", "fake")  # the kernel's llm()/rlm() use the rule-based fake
    ctx = tmp_path / "ctx.txt"
    ctx.write_text("alpha beta gamma\n" * 50)
    code = "a = llm('summarise: ' + ctx[:40])\nb = rlm('how long is your ctx?', ctx[:100])\nprint(a, b)"
    fb = FakeBackend([cell(code), cell("FINAL(b)")])
    r = run_agent("q", ctx_path=ctx, backend=fb, home=tmp_path)
    assert r["answer"] == "100"  # child agent computed len(sub_ctx) in its own kernel
    assert r["depth_used"] == 1 and r["subcalls"] == 2  # one llm() and one rlm() child, both at depth 1
    tree = "\n".join(render_tree(r["log_path"]))
    assert re.search(r"└─ llm\(\) \[in=\d+ out=\d+\]", tree)
    assert re.search(r"└─ rlm\(\) \[in=\d+ out=\d+\]", tree)
    assert "(depth 1)" in tree  # the child session is nested in the tree
    child = [e for e in load(r["log_path"]) if e.get("sub") == "rlm"][0]["child"]
    assert (tmp_path / "runs" / ("%s.jsonl" % child)).exists()  # child has its own log


def test_kernel_timeout_interrupts_and_kernel_survives(tmp_path):
    k = Kernel(cwd=tmp_path, timeout=1)
    try:
        k.execute("x = 41")
        out = k.execute("import time; time.sleep(30)")
        assert "timed out" in out
        assert k.execute("print(x + 1)").strip() == "42"
    finally:
        k.shutdown()


def test_cli_ask_trace_check_logs(tmp_path, monkeypatch, capsys):
    from rlm.cli import main
    monkeypatch.setenv("RLM_HOME", str(tmp_path))
    assert main(["--backend", "fake", "ask", "how long is ctx?"]) == 0
    out = capsys.readouterr()
    sid = re.search(r"session=(\w+)", out.err).group(1)
    assert main(["trace", sid]) == 0 and "FINAL" in capsys.readouterr().out
    assert main(["check-logs"]) == 0 and "PASS" in capsys.readouterr().out
