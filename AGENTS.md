# ONEFILE — Agent Instructions

ONEFILE is a single-file (`one.py`) crypto perpetual research project using vectorbt. Work autonomously, make the smallest complete change, and finish. Keep the model's reasoning and context intact; no new harness, judge, framework, or mandatory planning ritual.

## Fast development

- Locate the relevant functions with `rg`; read their callers and tests. Read the whole file only when the task requires it. Batch independent reads/commands in one tool call. Explain briefly, implement directly; do not repeatedly plan, reread, or audit unrelated code.
- Use the existing `.venv` with `uv run --no-sync`. Run `uv sync --frozen` only for initial setup or changed dependencies. `vectorbt[rust]` avoids Numba compilation on supported paths; keep the lockfile. Do not disable JIT globally or replace vectorbt with a custom engine.
- **No automatic test after edits and no baseline full-suite run for routine tasks.** Finish related edits first, then run the smallest check that covers the changed behavior once. For Python, use `uv run --no-sync python -m unittest discover -s tests -k ClassName` (or `-k test_method`). Batch multiple `-k` selections in one process. Docs/instruction-only changes need no Python tests.
- The local feedback target is **<=5 seconds**. If a check is unexpectedly slow, inspect `--durations 10` and narrow it to the affected behavior. Do not impose a timeout that hides failures, skip relevant assertions, or reduce research/bootstrap settings to hit the target. Report actual duration; cold installs and full research are separate costs.
- Run the full suite only for broad/cross-cutting changes, explicit requests, or CI: `uv run --no-sync python -m unittest discover -s tests`. It includes the 38-check execution self-test; do not also run `one.py --self-test`. Execution-only changes can use that self-test alone. Pi-only changes use `node --experimental-strip-types --test .pi/tests/onefile-extensions.test.mjs`.
- A passing check remains valid until relevant code, tests, dependencies, or inputs change. Do not rerun it for reassurance or because a tool hook also asks. After a failure, fix the cause and rerun the affected check; broaden only for an unresolved risk. `/onefile-verify` is optional and manual, never an automatic completion gate.

## Research integrity

Use DEV data; never unlock sealed OOS (`--unlock-oos`) without explicit user authorization. Avoid leakage/overfitting; preserve fees, slippage, funding, causal timing and statistical validation. Reuse valid data and evidence; do not re-download or rerun experiments without a concrete reason. Synthetic tests do not prove market edge, and historical evidence does not verify changed code.

Finish concisely: what changed, what actually passed (with time when measured), and anything unverified. Do not claim coding tasks themselves take <=5 seconds without measuring real model runs.
