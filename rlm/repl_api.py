"""Runs inside the kernel: defines ctx, grep, chunk, llm, rlm (+ rlm.spawn, rlm.harness) and FINAL."""
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SUB_SYSTEM = "You are a precise reading assistant. Answer only from the text you are given, briefly."
LLM_PROMPT_CAP = 40000  # chars per llm() prompt
FINAL_OPEN, FINAL_CLOSE = "<<RLM_FINAL>>", "<<RLM_END>>"


def grep(text, pattern, flags=0):
    rx = re.compile(pattern, flags)
    return [line for line in text.splitlines() if rx.search(line)]


def chunk(text, n):
    return [text[i:i + n] for i in range(0, len(text), int(n))]


def FINAL(answer):
    print("%s%s%s" % (FINAL_OPEN, answer, FINAL_CLOSE))


def install(g, cfg):
    from .backends import get_backend
    from .harness import Harness
    from .trace import SessionLog

    log = SessionLog(cfg["log_path"])
    backend = get_backend()
    pool = ThreadPoolExecutor(max_workers=4)

    def llm(prompt):
        prompt = str(prompt)[:LLM_PROMPT_CAP]
        text, usage = backend.complete(SUB_SYSTEM, [{"role": "user", "content": prompt}], role="sub")
        log.event("subcall", sub="llm", prompt=prompt[:300], prompt_chars=len(prompt), response=text[:300], tokens=usage)
        return text

    def rlm(prompt, sub_ctx=None):
        sub_ctx = g["ctx"] if sub_ctx is None else str(sub_ctx)
        if cfg["depth"] + 1 > cfg["max_depth"]:
            return llm("%s\n\n---\n%s" % (prompt, sub_ctx))  # depth limit: degrade to a plain sub-call
        from .agent import run_agent
        from .trace import new_id
        child = new_id()
        res = run_agent(str(prompt), ctx_text=sub_ctx, home=cfg["home"], runs_dir=Path(cfg["log_path"]).parent,
                        depth=cfg["depth"] + 1, parent=cfg["session"], session_id=child, backend=backend)
        log.event("subcall", sub="rlm", child=child, prompt=str(prompt)[:300], prompt_chars=len(sub_ctx),
                  response=str(res["answer"])[:300],
                  tokens={"input": res["tokens_in"], "output": res["tokens_out"], "model": "child"})
        return res["answer"]

    rlm.spawn = lambda prompt, sub_ctx=None: pool.submit(rlm, prompt, sub_ctx)
    rlm.harness = Harness(cfg["home"])
    g.update(ctx=Path(cfg["ctx_path"]).read_text(), grep=grep, chunk=chunk, llm=llm, rlm=rlm,
             harness=rlm.harness, FINAL=FINAL, re=re)
    for name, skill in rlm.harness.read("skills").items():
        exec(skill["code"], g)  # learned skills become kernel functions
