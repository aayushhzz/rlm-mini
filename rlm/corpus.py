"""Deterministic ~300k-char corpus (synthetic repo dump + 4 days of logs + operator notes) and 20 questions.

Gold answers are computed from the generated data, so they are exact.
"""
import random
from pathlib import Path

SERVICES = ["auth", "billing", "search", "gateway", "inventory", "notify"]
TEAMS = ["core-infra", "payments", "discovery", "edge", "supply", "messaging", "platform"]
PEOPLE = ["ana", "bo", "chen", "dara", "eli", "femi", "gus", "hana", "ivo", "jun", "kai", "lena"]
NEUTRAL_VENDORS = ["Kestrel Data", "Orbis Metrics", "Pallas CDN", "Quillon Mail", "Tessera Pay"]
LEVELS = [("INFO", 66), ("DEBUG", 10), ("WARN", 14), ("ERROR", 10)]
FUNCS = ["parse_request", "load_config", "rotate_keys", "flush_cache", "compute_score", "send_batch",
         "check_quota", "build_index", "sync_stock", "render_template", "validate_token", "retry_job"]
NOTE_TEMPLATES = [
    "Swapped on-call with {p} for the weekend, handoff doc is in the wiki.",
    "Renewed the {v} contract for another year, paperwork went smoothly.",
    "Reminder: {s} dashboards moved to the new folder, update your bookmarks.",
    "Pairing with {p} on the {s} flaky test tomorrow morning.",
    "{v} sent the quarterly usage report, numbers look in line with forecast.",
    "Lunch-and-learn on tracing next Thursday, {p} is presenting.",
    "Cleaned up stale feature branches in the {s} repo, about a dozen removed.",
    "Capacity review for {s} pushed to next week, {p} is out sick.",
    "Met with the {v} account manager, routine check-in, nothing to report.",
    "Rotated my laptop SSH keys after the security reminder email.",
]
VENDOR_COMPLAINT = ("Honestly fed up with Zentrix Cloud: third missed SLA this month and their support "
                    "keeps closing our tickets without a fix.")
LEAVE_NOTE = "Ravi starts parental leave on Monday; Ines will take over his on-call shifts until he is back."


def _ts(rng, day):
    return "2026-03-%02dT%02d:%02d:%02dZ" % (day, rng.randrange(24), rng.randrange(60), rng.randrange(60))


def build(seed=7):
    rng = random.Random(seed)
    facts = {"errors": {s: 0 for s in SERVICES}, "warns": {s: 0 for s in SERVICES}}

    # --- repo files -----------------------------------------------------------------------------
    files, facts["config"], facts["func_file"] = {}, {}, {}
    files["README.md"] = "# shopstack\n\nMonorepo for the shopstack services: %s.\n" % ", ".join(SERVICES) + \
        "Each service has services/<name>/config.yaml and handler.py. Logs are shipped to logs/app-<date>.log.\n"
    files["deploy/nodes.yaml"] = "".join("- name: edge-%d\n  region: %s\n  role: %s\n" % (
        i, rng.choice(["eu-west", "us-east", "ap-south"]), rng.choice(["edge", "worker"])) for i in range(1, 13))
    funcs = FUNCS[:] + ["reconcile_ledger"]
    rng.shuffle(funcs)
    for i, svc in enumerate(SERVICES):
        cfg = {"max_retries": rng.randint(2, 9), "timeout_ms": rng.randrange(500, 9000, 250),
               "pool_size": rng.randint(4, 64), "cache_ttl_s": rng.choice([30, 60, 120, 300, 600, 900]),
               "feature_flag": rng.choice(["alpha", "beta", "ga", "off"]), "owner_team": rng.choice(TEAMS)}
        facts["config"][svc] = cfg
        body = "# config for %s\nservice: %s\n" % (svc, svc)
        for k, v in cfg.items():
            body += "%s: %s\n" % (k, v)
        body += "".join("  # tuning note %d: revisit after load test\n" % j for j in range(3))
        files["services/%s/config.yaml" % svc] = body
        code = "import logging\nlog = logging.getLogger(%r)\n\n" % svc
        for fn in funcs[i * 2:i * 2 + 2] + (funcs[12:] if i == len(SERVICES) - 1 else []):
            facts["func_file"][fn] = "services/%s/handler.py" % svc
            code += ("def %s(payload, retries=3):\n    \"\"\"%s step for %s.\"\"\"\n    for attempt in range(retries):\n"
                     "        try:\n            return _do(payload)\n        except Exception as exc:\n"
                     "            log.error(\"%s failed: %%s\", exc)\n    return None\n\n") % (fn, fn, svc, fn)
        files["services/%s/handler.py" % svc] = code
    # Padding modules so the repo dump is realistically long.
    for j in range(30):
        files["lib/util_%02d.py" % j] = "".join(
            "def helper_%d_%d(x):\n    # pure helper, no I/O\n    return (x * %d + %d) %% 97\n\n" % (j, k, rng.randint(2, 50), k)
            for k in range(14))

    # --- logs -----------------------------------------------------------------------------------
    req_pool = ["req-%06x" % rng.randrange(16 ** 6) for _ in range(30)]
    used_503 = set()
    weights = {s: rng.uniform(0.6, 1.6) for s in SERVICES}
    days = {}
    for day in (1, 2, 3, 4):
        lines = []
        for _ in range(600):
            svc = rng.choices(SERVICES, [weights[s] for s in SERVICES])[0]
            lvl = rng.choices([l for l, _ in LEVELS], [w for _, w in LEVELS])[0]
            if lvl == "ERROR" and svc == "billing" and rng.random() < 0.5:
                rid = rng.choice(req_pool)
                used_503.add(rid)
                msg = "request %s failed status=503 upstream=ledger" % rid
            elif lvl == "ERROR":
                msg = "request req-%06x failed status=%d latency_ms=%d" % (rng.randrange(16 ** 6), rng.choice([500, 502, 504]), rng.randint(100, 9000))
            elif lvl == "WARN":
                msg = "slow response latency_ms=%d threshold_ms=%d" % (rng.randint(900, 5000), 800)
            elif lvl == "DEBUG":
                msg = "cache lookup key=%s hit=%s" % ("k%05d" % rng.randrange(99999), rng.choice(["true", "false"]))
            else:
                msg = "handled request req-%06x status=200 latency_ms=%d" % (rng.randrange(16 ** 6), rng.randint(5, 700))
            lines.append([_ts(rng, day), svc, lvl, msg])
        for _ in range(100):  # operator notes, free text: needs reading, not regex
            t = rng.choice(NOTE_TEMPLATES).format(p=rng.choice(PEOPLE), v=rng.choice(NEUTRAL_VENDORS), s=rng.choice(SERVICES))
            lines.append([_ts(rng, day), "ops-note", "NOTE", "author=%s: %s" % (rng.choice(PEOPLE), t)])
        days[day] = lines
    days[2].append([_ts(rng, 2), "billing", "ERROR", "INC-5821 opened, root cause: disk full on db-replica-3"])
    days[3].append([_ts(rng, 3), "ops-note", "NOTE", "author=mira: " + VENDOR_COMPLAINT])
    days[4].append([_ts(rng, 4), "ops-note", "NOTE", "author=tomas: " + LEAVE_NOTE])
    days[3].append([_ts(rng, 3), "gateway", "INFO", "deploy token rotated on node edge-7 by user priya.n"])
    days[4].append([_ts(rng, 4), "gateway", "INFO", "deploy token rotated on node edge-11 by user lmorales"])
    days[1].append([_ts(rng, 1), "gateway", "INFO", "deploy token rotated on node edge-2 by user okafor"])
    facts["inventory_day3"] = 0
    for day, lines in days.items():
        lines.sort(key=lambda r: r[0])
        for ts, svc, lvl, msg in lines:
            if svc in SERVICES:
                facts["errors"][svc] += lvl == "ERROR"
                facts["warns"][svc] += lvl == "WARN"
                facts["inventory_day3"] += svc == "inventory" and day == 3
    facts["distinct_503"] = len(used_503)

    # --- assemble: repo and logs interleaved so needles are spread across the 300k chars ---------
    def file_block(paths):
        return "".join("\n===== FILE: %s =====\n%s" % (p, files[p]) for p in paths)

    def log_block(day):
        return "\n===== LOG: logs/app-2026-03-%02d.log =====\n" % day + "".join(
            "%s [%s] %s %s\n" % tuple(r) for r in days[day])

    utils = sorted(p for p in files if p.startswith("lib/"))
    svc_files = lambda names: [p for n in names for p in ("services/%s/config.yaml" % n, "services/%s/handler.py" % n)]
    text = (file_block(["README.md", "deploy/nodes.yaml"] + svc_files(["auth", "billing"]) + utils[:10]) + log_block(1)
            + file_block(svc_files(["search", "gateway"]) + utils[10:20]) + log_block(2)
            + log_block(3) + file_block(utils[20:] + svc_files(["inventory", "notify"])) + log_block(4))
    return text, facts


def questions(facts):
    q = []
    for s in SERVICES:
        q.append(("agg", "How many ERROR log lines does the %s service have in total across all log files?" % s, str(facts["errors"][s])))
    q.append(("agg", "Which service has the most WARN log lines in total?", max(facts["warns"], key=facts["warns"].get)))
    q.append(("agg", "How many distinct request IDs failed with status=503 in the billing service?", str(facts["distinct_503"])))
    q.append(("agg", "How many log lines (any level) did the inventory service write on 2026-03-03?", str(facts["inventory_day3"])))
    for svc, key in [("notify", "timeout_ms"), ("auth", "max_retries"), ("inventory", "cache_ttl_s"),
                     ("gateway", "pool_size"), ("search", "feature_flag")]:
        q.append(("needle", "In the file services/%s/config.yaml, what is %s set to?" % (svc, key), str(facts["config"][svc][key])))
    q.append(("needle", "Which file defines the function reconcile_ledger?", facts["func_file"]["reconcile_ledger"]))
    q.append(("needle", "Which user rotated the deploy token on node edge-7?", "priya.n"))
    q.append(("needle", "On which node did user lmorales rotate the deploy token?", "edge-11"))
    q.append(("needle", "What is the incident ID whose root cause was a full disk on db-replica-3?", "INC-5821"))
    q.append(("semantic", "In the free-text operator notes, one vendor is the target of a complaint. Which vendor?", "Zentrix"))
    q.append(("semantic", "According to the operator notes, who will take over Ravi's on-call shifts?", "Ines"))
    return [{"id": "q%02d" % (i + 1), "type": t, "question": text, "gold": g} for i, (t, text, g) in enumerate(q)]


def ensure_corpus(home):
    """Write corpus/corpus.txt under home (idempotent) and return (path, questions)."""
    text, facts = build()
    path = Path(home) / "corpus" / "corpus.txt"
    if not path.exists() or path.read_text() != text:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return path, questions(facts)
