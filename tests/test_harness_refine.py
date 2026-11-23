"""Continual Harness: one minimal edit per /refine, byte-identical rollback, base prompt never modified."""
import hashlib
import json

import pytest

from rlm.backends import FakeBackend
from rlm.harness import FILES, Harness
from rlm.refine import refine
from rlm.trace import SessionLog

EVIDENCE = "NameError: name 'lines' is not defined"


def _trajectory(tmp_path):
    log = SessionLog(tmp_path / "runs" / "abc123.jsonl")
    log.event("session_start", session="abc123", parent=None, depth=0, question="How many ERROR lines for auth?",
              ctx_path=None, ctx_chars=100, backend="fake")
    log.event("cell", turn=1, code="print(len(lines))", output=EVIDENCE)
    log.event("final", answer="12")
    return log.path


def _edit(**kw):
    e = {"file": "memory", "op": "create", "key": "log_format",
         "value": "Log lines look like '<ts> [service] LEVEL msg'; split ctx with ctx.splitlines() first.",
         "evidence": EVIDENCE, "rationale": "the model used an undefined variable"}
    e.update(kw)
    return json.dumps(e)


def _files(h):
    return {f: h.path(f).read_bytes() for f in FILES}


def _sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_refine_writes_exactly_one_edit_with_snapshot_and_evidence(tmp_path):
    h = Harness(tmp_path)
    h.path("prompts").write_bytes(b'{"keep":   "odd  spacing kept"}\n')  # non-canonical bytes on purpose
    before = _files(h)
    sid, edit, _ = refine(tmp_path, FakeBackend([_edit()]), log_path=_trajectory(tmp_path))
    after = _files(h)
    assert [f for f in FILES if before[f] != after[f]] == ["memory"]  # one file changed
    assert json.loads(after["memory"]) == {"log_format": json.loads(_edit())["value"]}  # one key added
    assert len(h.snapshots()) == 1
    meta = h.meta(sid)
    assert meta["evidence"] == EVIDENCE and meta["session"] == "abc123"
    assert meta["trigger"] and "+  \"log_format\"" in meta["diff"]


def test_rollback_restores_bytes_exactly(tmp_path):
    h = Harness(tmp_path)
    h.path("memory").write_bytes(b'{\n "existing":"x"   }\n')
    before = _files(h)
    sid, _, _ = refine(tmp_path, FakeBackend([_edit()]), log_path=_trajectory(tmp_path))
    assert _files(h) != before
    h.rollback(sid)
    assert _files(h) == before
    assert "rolled back" in h.meta(sid)["outcome"]


def test_rollback_of_update_and_delete(tmp_path):
    h = Harness(tmp_path)
    s1 = h.create("prompts", "p1", "prefer grep over manual scanning")
    mid = _files(h)
    s2 = h.update("prompts", "p1", "prefer grep, then verify counts")
    s3 = h.delete("prompts", "p1")
    assert h.read("prompts") == {}
    h.rollback(s3)
    assert h.read("prompts") == {"p1": "prefer grep, then verify counts"}
    h.rollback(s2)
    assert _files(h) == mid
    h.rollback(s1)
    assert h.path("prompts").read_bytes() == b"{}\n"


def test_refine_never_modifies_base_system_prompt(tmp_path):
    h = Harness(tmp_path)
    base_sha = _sha(h.base_path)
    traj = _trajectory(tmp_path)
    # The model first tries to rewrite the base prompt, then sends a bad-evidence edit, then a valid one.
    script = [_edit(file="base_system_prompt", op="update", key="all", value="ignore rules"),
              _edit(evidence="this text is not in the trajectory"),
              _edit()]
    sid, edit, _ = refine(tmp_path, FakeBackend(script), log_path=traj)
    assert edit["file"] == "memory"
    assert _sha(h.base_path) == base_sha
    for _ in range(3):  # more rounds, still untouched
        refine(tmp_path, FakeBackend([_edit(key="k%d" % _)]), log_path=traj)
    assert _sha(h.base_path) == base_sha
    assert h.system_prompt().startswith(h.base_path.read_text().rstrip())


def test_refine_rejects_leaked_gold_answer_and_bad_skill(tmp_path):
    h = Harness(tmp_path)
    traj = _trajectory(tmp_path)
    with pytest.raises(ValueError):
        refine(tmp_path, FakeBackend([_edit(value="auth has 12 ERROR lines")] * 3), log_path=traj, forbidden=["12"])
    with pytest.raises(ValueError):
        refine(tmp_path, FakeBackend([_edit(file="skills", value={"description": "d", "code": "def f(:"})] * 3), log_path=traj)
    assert h.snapshots() == [] and _files(h) == {f: b"{}\n" for f in FILES}


def test_skill_is_defined_in_system_prompt(tmp_path):
    h = Harness(tmp_path)
    h.create("skills", "count_level", {"description": "count lines of a level for a service",
                                       "code": "def count_level(svc, lvl):\n    return len(grep(ctx, r'\\[%s\\] %s ' % (svc, lvl)))\n"})
    assert "count_level: count lines of a level" in h.system_prompt()
