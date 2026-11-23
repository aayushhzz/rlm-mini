# rlm bench results

- bench id: `20261120-124206` (logs in `bench/20261120-124206/runs/`), backend: `claude`, questions: 2, wall time: 509s
- stuffing arm: first 80000 chars of the 294,698-char corpus (32k-token window), one call per question
- refine tokens (not in the arm totals): 15319

| Arm | Accuracy | Total tokens | Mean tokens / question | Max recursion depth | Sub-calls (llm/rlm) |
|---|---|---|---|---|---|
| (a) prompt stuffing, 32k window | 0/2 (0%) | 80,695 | 40,347 | 0 | 0 |
| (b) RLM | 2/2 (100%) | 52,575 | 26,287 | 0 | 0 |
| (c) RLM + 3 refines | 2/2 (100%) | 261,923 | 130,961 | 0 | 0 |

## Per question

| id | type | gold | stuffing | RLM | RLM+refine | RLM session |
|---|---|---|---|---|---|---|
| q19 | semantic | `Zentrix` | x `None` | ok `Zentrix Cloud` | ok `Zentrix Cloud` | `8ce2d1f1f2` |
| q20 | semantic | `Ines` | x `The document doesn't say. Ra` | ok `Ines` | ok `Ines` | `e8cdb38efc` |

## Refine edits (Continual Harness snapshots)

- `s0001` from q19: create `memory/operator_notes_format` = Free-text operator notes live in the log files as lines containing '[ops-note] NOTE author=<name>: <body>'. Most bodies are routine boilerplate (SSH key rotatio  
  evidence: "7 #-#-#T#:#:#Z [ops-note] NOTE author=ana: Cleaned up stale feature branches in th"  
  rationale: The trajectory spent several turns locating the notes by guessing at file headers and keywords like 'operator' and 'complain', which matched nothing. A short me
- `s0002` from q20: create `memory/operator_notes_coverage` = Notes about a person's leave or absence name the replacement in the note body, not the author (e.g. 'X will take over Y's on-call shifts'). Routine 'Swapped on-  
  evidence: "Ravi starts parental leave on Monday; Ines will take over his on-call shifts until he is back."  
  rationale: The trajectory found the answer only after a narrow name grep, and the many routine 'Swapped on-call' decoys made the broad on-call search noisy. A short note o
- `s0003` from q19: create `skills/ops_notes_nonroutine` = {'description': "Return (author, body) pairs for '[ops-note]' lines in ctx whose body is not routine boilerplate, so substantive notes such as vendor complaints  
  evidence: "#-#-#T#:#:#Z [ops-note] NOTE author=ana: Cleaned up stale feature branches in th"  
  rationale: The trajectory spent several turns re-deriving the same routine-versus-substantive filter over '[ops-note]' lines before reaching the complaint. A reusable skil
