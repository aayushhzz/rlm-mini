"""Every model call goes through one interface: complete(system, messages, role, num_ctx) -> (text, usage).

role is "root" (the code-writing model) or "sub" (llm() sub-calls). usage = {"input", "output", "model"}.
"""
import json
import os
import subprocess
import tempfile
import urllib.request


def _render(messages):
    """Flatten a chat into one prompt for the stateless `claude -p` call."""
    if len(messages) == 1:
        return messages[0]["content"]
    parts = []
    for m in messages:
        tag = "ASSISTANT (your earlier turn)" if m["role"] == "assistant" else "USER"
        parts.append("=== %s ===\n%s" % (tag, m["content"]))
    parts.append("=== Write your next ASSISTANT turn now. ===")
    return "\n\n".join(parts)


class ClaudeCLIBackend:
    """Claude via the locally logged-in Claude Code CLI (the user's subscription), no API key."""

    name = "claude"

    def __init__(self, timeout=600):
        self.timeout = timeout
        self.models = {"root": os.environ.get("RLM_MODEL"), "sub": os.environ.get("RLM_SUB_MODEL")}
        # Low effort by default: one runaway thinking turn can burn >100k output tokens of Pro usage.
        self.effort = os.environ.get("RLM_EFFORT", "low")

    def complete(self, system, messages, role="root", num_ctx=None):
        cmd = ["claude", "-p", "--output-format", "json", "--system-prompt", system,
               "--tools", "", "--no-session-persistence", "--setting-sources", "",
               "--strict-mcp-config", "--disable-slash-commands"]
        model = self.models.get(role) or self.models.get("root")
        if model:
            cmd += ["--model", model]
        if self.effort:
            cmd += ["--effort", self.effort]
        # Drop any API key so the CLI always uses the claude.ai login.
        env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
        with tempfile.TemporaryDirectory() as cwd:  # empty cwd: no project CLAUDE.md gets loaded
            p = subprocess.run(cmd, input=_render(messages), capture_output=True, text=True,
                               cwd=cwd, env=env, timeout=self.timeout)
        try:
            out = json.loads(p.stdout)
        except ValueError:
            raise RuntimeError("claude CLI failed (%s): %s" % (p.returncode, (p.stderr or p.stdout)[:500]))
        if out.get("is_error"):
            raise RuntimeError("claude CLI error: %s" % str(out.get("result"))[:500])
        u = out.get("usage", {})
        inp = u.get("input_tokens", 0) + u.get("cache_creation_input_tokens", 0) + u.get("cache_read_input_tokens", 0)
        mu = out.get("modelUsage") or {}
        used = "+".join(sorted(mu, key=lambda k: -mu[k].get("outputTokens", 0))) or model or "default"
        return out.get("result") or "", {"input": inp, "output": u.get("output_tokens", 0), "model": used}


class OllamaBackend:
    """Optional local fallback: qwen3:8b root, qwen3:4b-instruct for llm() sub-calls.

    Uses Ollama's native /api/chat because the OpenAI-compatible endpoint cannot set num_ctx per request.
    """

    name = "ollama"

    def __init__(self, timeout=900):
        self.url = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/") + "/api/chat"
        self.timeout = timeout
        self.models = {"root": os.environ.get("RLM_MODEL") or "qwen3:8b",
                       "sub": os.environ.get("RLM_SUB_MODEL") or "qwen3:4b-instruct"}

    def complete(self, system, messages, role="root", num_ctx=None):
        model = self.models[role]
        body = {"model": model, "stream": False,
                "messages": [{"role": "system", "content": system}] + list(messages),
                "options": {"num_ctx": num_ctx or 16384, "temperature": 0}}
        if "instruct" not in model:
            body["think"] = False  # qwen3 hybrid models: skip the long thinking trace
        req = urllib.request.Request(self.url, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            out = json.loads(r.read())
        return out["message"]["content"], {"input": out.get("prompt_eval_count", 0),
                                           "output": out.get("eval_count", 0), "model": model}


class FakeBackend:
    """Offline backend for tests: pops scripted replies, then falls back to a fixed rule."""

    name = "fake"

    def __init__(self, script=None):
        self.script = list(script or [])
        self.calls = []

    def complete(self, system, messages, role="root", num_ctx=None):
        self.calls.append({"system": system, "messages": [dict(m) for m in messages], "role": role})
        if self.script:
            text = self.script.pop(0)
        elif role == "sub":
            text = "fake-sub:%d" % len(messages[-1]["content"])
        else:
            text = "```python\nFINAL(len(ctx))\n```"
        n_in = (len(system) + sum(len(m["content"]) for m in messages)) // 4
        return text, {"input": n_in, "output": len(text) // 4, "model": "fake"}


def get_backend(name=None):
    name = name or os.environ.get("RLM_BACKEND", "claude")
    if name == "claude":
        return ClaudeCLIBackend()
    if name == "ollama":
        return OllamaBackend()
    if name == "fake":
        return FakeBackend()
    raise ValueError("unknown RLM_BACKEND %r (claude | ollama | fake)" % name)
