"""Continual Harness: CRUD over harness_state/{prompts,skills,memory}.json with a snapshot per edit.

The base system prompt (harness_state/base_system_prompt.md) is read here but never written.
"""
import difflib
import hashlib
import json
import shutil
import time
from pathlib import Path

from . import PACKAGE_DIR

FILES = ("prompts", "skills", "memory")
BASE_NAME = "base_system_prompt.md"
MAX_VALUE_CHARS = 1500


class Harness:
    def __init__(self, home):
        self.dir = Path(home) / "harness_state"
        self.snap_dir = self.dir / "snapshots"
        self.snap_dir.mkdir(parents=True, exist_ok=True)
        for f in FILES:
            if not self.path(f).exists():
                self.path(f).write_text("{}\n")
        if not self.base_path.exists():
            shutil.copyfile(PACKAGE_DIR / BASE_NAME, self.base_path)

    @property
    def base_path(self):
        return self.dir / BASE_NAME

    def path(self, f):
        if f not in FILES:
            raise ValueError("harness file must be one of %s, not %r" % (FILES, f))
        return self.dir / ("%s.json" % f)

    def read(self, f):
        return json.loads(self.path(f).read_text())

    # CRUD; every write goes through apply_edit so it is snapshotted.
    def create(self, f, key, value, **why):
        return self.apply_edit({"file": f, "op": "create", "key": key, "value": value}, **why)

    def update(self, f, key, value, **why):
        return self.apply_edit({"file": f, "op": "update", "key": key, "value": value}, **why)

    def delete(self, f, key, **why):
        return self.apply_edit({"file": f, "op": "delete", "key": key}, **why)

    def validate(self, edit):
        f, op, key = edit.get("file"), edit.get("op"), edit.get("key")
        data = self.read(f)  # raises for anything outside FILES, base prompt included
        if op not in ("create", "update", "delete"):
            raise ValueError("op must be create|update|delete")
        if not isinstance(key, str) or not key.strip() or len(key) > 64:
            raise ValueError("key must be a non-empty string of at most 64 chars")
        if op == "create" and key in data:
            raise ValueError("%s/%s already exists; use update" % (f, key))
        if op in ("update", "delete") and key not in data:
            raise ValueError("%s/%s does not exist" % (f, key))
        if op == "delete":
            return
        value = edit.get("value")
        if f == "skills":
            if not (isinstance(value, dict) and isinstance(value.get("code"), str)):
                raise ValueError("a skill value is {\"description\": str, \"code\": str}")
            try:
                compile(value["code"], "<skill %s>" % key, "exec")
            except SyntaxError as e:
                raise ValueError("skill code does not compile: %s" % e)
            size = len(value["code"]) + len(str(value.get("description", "")))
        else:
            if not isinstance(value, str) or not value.strip():
                raise ValueError("prompts/memory values are non-empty strings")
            size = len(value)
        if size > MAX_VALUE_CHARS:
            raise ValueError("value is %d chars; the limit is %d" % (size, MAX_VALUE_CHARS))

    def apply_edit(self, edit, trigger="agent", evidence="", rationale="", session=None):
        self.validate(edit)
        f, key = edit["file"], edit["key"]
        p = self.path(f)
        before = p.read_bytes()
        data = self.read(f)
        if edit["op"] == "delete":
            del data[key]
        else:
            data[key] = edit["value"]
        after = (json.dumps(data, indent=2, sort_keys=True) + "\n").encode()
        sid = self._next_id()
        snap = self.snap_dir / sid
        snap.mkdir()
        (snap / ("%s.json.before" % f)).write_bytes(before)
        diff = "".join(difflib.unified_diff(before.decode().splitlines(True), after.decode().splitlines(True),
                                            "a/%s.json" % f, "b/%s.json" % f))
        meta = {"id": sid, "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "trigger": trigger,
                "session": session, "edit": edit, "evidence": evidence, "rationale": rationale,
                "file": f, "sha256_before": hashlib.sha256(before).hexdigest(),
                "sha256_after": hashlib.sha256(after).hexdigest(), "diff": diff, "outcome": "applied"}
        (snap / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        p.write_bytes(after)
        return sid

    def _next_id(self):
        n = len([d for d in self.snap_dir.iterdir() if d.is_dir()]) + 1
        return "s%04d" % n

    def meta(self, sid):
        return json.loads((self.snap_dir / sid / "meta.json").read_text())

    def set_outcome(self, sid, outcome):
        m = self.meta(sid)
        m["outcome"] = outcome
        (self.snap_dir / sid / "meta.json").write_text(json.dumps(m, indent=2) + "\n")

    def rollback(self, sid):
        """Restore the edited file to its exact pre-edit bytes."""
        m = self.meta(sid)
        shutil.copyfile(self.snap_dir / sid / ("%s.json.before" % m["file"]), self.path(m["file"]))
        self.set_outcome(sid, "rolled back %s" % time.strftime("%Y-%m-%dT%H:%M:%S"))
        return m

    def snapshots(self):
        return [self.meta(d.name) for d in sorted(self.snap_dir.iterdir()) if (d / "meta.json").exists()]

    def system_prompt(self):
        """Immutable base prompt plus the supplemental Continual Harness state."""
        parts = [self.base_path.read_text().rstrip()]
        prompts, memory, skills = self.read("prompts"), self.read("memory"), self.read("skills")
        if prompts:
            parts.append("## Prompt notes (learned)\n" + "\n".join("- %s: %s" % kv for kv in sorted(prompts.items())))
        if memory:
            parts.append("## Memory (learned facts)\n" + "\n".join("- %s: %s" % kv for kv in sorted(memory.items())))
        if skills:
            parts.append("## Skills (already defined in the kernel)\n" + "\n".join(
                "- %s: %s" % (k, v.get("description", "")) for k, v in sorted(skills.items())))
        return "\n\n".join(parts) + "\n"
