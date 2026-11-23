import re

from rlm.backends import FakeBackend
from rlm.bench import run_bench, score
from rlm.corpus import ensure_corpus


def test_score_is_whole_token():
    assert score("There are 4 errors.", "4")
    assert not score("14", "4")
    assert score("**Zentrix Cloud**", "Zentrix")
    assert score("`services/notify/handler.py`", "services/notify/handler.py")
    assert not score("edge-110", "edge-11")


def test_corpus_golds_match_the_text(tmp_path):
    path, qs = ensure_corpus(tmp_path)
    text = path.read_text()
    assert 280000 < len(text) < 320000 and len(qs) == 20
    gold = {q["id"]: q["gold"] for q in qs}
    for i, svc in enumerate(["auth", "billing", "search", "gateway", "inventory", "notify"]):
        assert len(re.findall(r"^\S+ \[%s\] ERROR " % svc, text, re.M)) == int(gold["q%02d" % (i + 1)])
    assert re.search(r"edge-7 by user priya\.n", text)
    block = text.split("===== FILE: services/notify/config.yaml =====")[1]
    assert "timeout_ms: %s\n" % gold["q10"] in block.split("=====")[0]
    assert text.index("Zentrix") > 80000  # semantic needle lies beyond the stuffing window


def test_bench_writes_three_arm_table_offline(tmp_path, monkeypatch):
    monkeypatch.setenv("RLM_BACKEND", "fake")
    edit = ('{"file": "prompts", "op": "create", "key": "verify_%d", "value": "Re-check counts with a '
            'second method before FINAL.", "evidence": "FINAL ANSWER:", "rationale": "unverified"}')
    # stuffing x2, RLM x2 (the first trajectory makes an llm() and an rlm() sub-call), three refine edits;
    # arm (c) then uses the fake's fallback rule
    sub = "```python\na = llm('count?')\nb = rlm('len?', ctx[:50])\nFINAL(64)\n```"
    script = ["64", "37", sub, "FINAL: 3"] + [edit % i for i in range(3)]
    md = run_bench(tmp_path, FakeBackend(script), limit=2, rounds=3)
    assert "| (a) prompt stuffing, first 80k chars | 1/2 (50%)" in md
    assert re.search(r"\| \(b\) RLM \| 1/2 \(50%\) \| [\d,]+ \| [\d,]+ \| 1 \| 2 \|", md)  # depth 1, 2 sub-calls
    assert "| (c) RLM + 3 refines | 0/2" in md
    assert (tmp_path / "results.md").read_text() == md
    snaps = sorted((tmp_path / "bench").glob("*/harness_state/snapshots/s000*"))
    assert len(snaps) == 3
