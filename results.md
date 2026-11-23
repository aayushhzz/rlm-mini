# rlm bench results

- bench id: `20261120-123045` (logs in `bench/20261120-123045/runs/`), backend: `claude`, questions: 5, wall time: 319s
- stuffing arm: first 80000 chars of the 294,698-char corpus (32k-token window), one call per question
- refine tokens (not in the arm totals): 10081

| Arm | Accuracy | Total tokens | Mean tokens / question | Max recursion depth | Sub-calls (llm/rlm) |
|---|---|---|---|---|---|
| (a) prompt stuffing, 32k window | 0/5 (0%) | 202,112 | 40,422 | 0 | 0 |
| (b) RLM | 5/5 (100%) | 88,093 | 17,618 | 0 | 0 |
| (c) RLM + 3 refines | 5/5 (100%) | 86,248 | 17,249 | 0 | 0 |

## Per question

| id | type | gold | stuffing | RLM | RLM+refine | RLM session |
|---|---|---|---|---|---|---|
| q01 | agg | `64` | x `13` | ok `64` | ok `64` | `3128ad75e7` |
| q08 | agg | `17` | x `6` | ok `17` | ok `17` | `6019742216` |
| q10 | needle | `2250` | x `Not specified in the documen` | ok `2250` | ok `2250` | `4c1dd6c6aa` |
| q19 | semantic | `Zentrix` | x `None` | ok `Zentrix Cloud` | ok `Zentrix Cloud` | `98dbf2eb60` |
| q20 | semantic | `Ines` | x `The document doesn't say. No` | ok `Ines` | ok `Ines` | `24d92311e5` |

## Refine edits (Continual Harness snapshots)

- `s0001` from q01: create `memory/log_line_format` = Log lines look like '<ISO timestamp> [<service>] <LEVEL> ...'. Logs for a service are not confined to its own services/<name>/ file, so count a service's entrie  
  evidence: "2026-03-01T00:02:08Z [auth] ERROR request req-b091f8 failed status=500 latency_ms=570"  
  rationale: The agent first counted by file, then found [auth] ERROR lines spread across lib/util_09.py, lib/util_19.py, and services/notify/handler.py. Recording the line 
- `s0002` from q08: create `memory/status_field_format` = HTTP status appears as 'status=<code>' (e.g. 'status=503'). Other numbers such as 'latency_ms=503' can match a bare '503', so anchor status searches on 'status=  
  evidence: "2026-03-01T12:28:49Z [auth] INFO handled request req-d7aad8 status=200 latency_ms=503"  
  rationale: The first grep for '503' returned 27 lines, and turn 2 showed 5 of them were latency_ms=503 false positives. A memory note on the status field format prevents t
- `s0003` from q10: create `memory/file_block_format` = The context concatenates files, each introduced by a header line '===== FILE: <path> ====='. To read one file, locate its header and slice up to the next '\n===  
  evidence: "===== FILE: services/notify/config.yaml ====="  
  rationale: The agent had to work out the file-block boundaries by hand. Recording the header format lets future questions about any file in ctx extract its block directly,
