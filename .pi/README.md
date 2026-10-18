# ONEFILE / Pi 1.3.0 — Latency-first setup

Scope: **only `one.py` is the Python app**. Pi customizations live here and do not change the quantitative engine. This directory is a project-local Pi configuration; it becomes active after the repo has been trusted by the Pi operator.

## Active in this repository

- `settings.json`: enable Pi's native **Code Mode (`on`)** alongside regular tools; do not force scripts for every simple action.
- `extensions/onefile-core.ts`: warn on old source/evidence SHA, reject ordinary tool calls containing `--unlock-oos`, and block a third identical failed shell command against unchanged `one.py`. `/onefile-status` shows hashes.
- `extensions/onefile-verify.ts`: debounce background smoke checks after direct `edit` or `write` of `one.py` / `pyproject.toml` / `uv.lock`; at final settlement, check the **current** revision again if code/dependencies changed. `/onefile-verify` runs both checks on demand. Checks are `uv run --no-sync python -m py_compile one.py` and `uv run --no-sync python one.py --self-test`.
- `../AGENTS.md`: concise scientific and development rules. It should not instruct the agent to reduce reasoning quality or context.

**Before use:** in your own environment, run `uv sync --frozen` once so `--no-sync` works. Keep model/API credentials outside Git. Read third-party extension code before installing it. Use `pi config --local` to inspect effective resources. `pi list` helps identify duplicates.

The background check does not run in a frozen filesystem snapshot. Its hash-bound results are discarded if the source or lockfile changes while tests run; the final check never accepts an older revision. This is a latency experiment, **not** a secure isolation layer. Concurrent writers may interfere with each other's running tests. Use an isolated worktree/session for concurrent agents.

## External extensions: *recommended*, **not installed by this commit**

| Candidate | Source | Status / condition |
| --- | --- | --- |
| Pi Trace | https://github.com/npxcnency-ux/pi-trace-extension | Local JSONL/HTML traces. Trial in an isolated Pi 1.3.0 environment; beware raw prompts/secrets in trace files, ensure they never enter Git. |
| pi-search | https://github.com/justhil/pi-search | On-demand official, version-matched Python/vectorbt/Binance docs. Configure its search provider credentials first; not a mandatory startup gate. |
| SoL-Pi Action Fusion | https://github.com/NVlabs/SoL-Pi | **Not enabled yet.** Its public compatibility report tests Pi 0.84.2–0.85.1, not 1.3.0. Independently test *only* Action Fusion; leave ObservationPack, reducer and online compaction off. Avoid duplicating the verification hook. |

Pin the tested npm or Git version/commit in the **local Pi installation** after confirming compatibility. Do not vendor upstream packages into ONEFILE or declare them installed prematurely.

## Scientific safety and limitations

- OOS data must be kept **outside the agent-accessible checkout**, in a separately controlled environment. `onefile-core.ts` blocks the literal CLI flag in ordinary Pi tools and their Code Mode nested calls by default. It is **not a sandbox**: alternate shell syntax, other programs, external tools, direct user shell access, or already-mounted files can bypass it. Never rely on regex guarding to seal OOS data. Human opt-in for a separate authorized run is `ONEFILE_ALLOW_OOS=1` set **before starting Pi**; do not configure this for routine DEV sessions.
- The committed `brick1_evidence.json` may describe a previous source revision; use `/onefile-status`. The extension does not edit evidence or certify historical research.
- `--self-test` is **synthetic regression testing** using vectorbt. It does not establish edge, valid OOS, or a runnable $500 live strategy. Evidence needs source/data SHA, dependencies, experiment settings and independent evaluation.
- Background/final checks fail closed if locked dependencies are missing; they do not auto-install or alter the research environment.
- Trace results should be compared on **successful coding tasks**, same model/provider/effort and same starting SHA. Do not claim 20→5 minutes without real measured runs.

## Local extension tests

`node --experimental-strip-types --test .pi/tests/onefile-extensions.test.mjs`

CI runs these deterministic hook checks independently of the Python self-tests. They are not real Pi model calls or actual provider performance benchmarks. A separate Pi 1.3.0 smoke test is still required on the intended VPS.

## Deliberately excluded

Lean-CTX, Jev/judge models, context pruning, ObservationPack, forced subagents, speculative file prefetch, an external DAG scheduler and a custom harness/fork. Revisit only after instrumented traces demonstrate a critical-path need.