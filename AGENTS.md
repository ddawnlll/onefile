# ONEFILE — Agent Contract

## Goal
ONEFILE is a single-file, evidence-first research project for crypto perpetual futures. Discover and falsify economic mechanisms. A negative result is valid; a profitable backtest alone is not proof of edge.

## Scope and speed
- `one.py` is the only Python application source. Keep it that way; use the existing vectorbt execution engine, not another backtester.
- Read `one.py` in full when a task requires understanding the system. Do not repeatedly reread unchanged content; use targeted reads for subsequent edits.
- Reuse current functions and dependencies. Prefer the smallest clear implementation over new abstractions, agents, repositories, or frameworks.
- Use independent tool operations in parallel where safe. Code Mode is optional, not mandatory for simple edits. Do not reduce reasoning effort or withhold needed context to save tokens.
- Search official, version-matched vectorbt/Python/Binance docs when behavior is uncertain. Do not invent API contracts or perform irrelevant browsing.

## Research validity
- Keep routine investigation and iteration strictly inside the DEV period (`2020-01-01` through `2024-12-31`). **Do not run `--unlock-oos`, access sealed OOS data, or inspect OOS outcomes without the user's explicit authorization for that separate evaluation.** The guard is only a defense against accidental tool calls, not a secure sandbox; keep OOS files outside the agent's reachable workspace.
- Prevent lookahead, data leakage, multiple-testing abuse, overfitting and silent changes to historical data. Preserve conservative execution assumptions, realistic fees/slippage/funding and month-matched controls.
- Distinguish synthetic self-tests, descriptive DEV findings and independently validated OOS evidence. `EXPLORATORY PASS` is not proof of a tradable edge.
- Never treat evidence recorded for an older source/data hash as proof about the current checkout. Keep original evidence immutable where possible; emit new versioned artifacts for new research runs.

## Verification
- For changes to `one.py` or its locked environment run `uv run --no-sync python -m py_compile one.py` and `uv run --no-sync python one.py --self-test` after `uv sync --frozen` has prepared the environment.
- These checks use synthetic fixtures; they do **not** establish market edge, statistical validity or OOS performance. Real DEV experiments need separately recorded data SHA-256, source SHA-256, versions, parameters and command output.
- Associate test results with the final source/dependency revision. Ignore stale results from earlier edits, and never skip a failing test to declare success.
- If the same command fails repeatedly without any relevant change, diagnose before retrying. When running tests in the background, do not use results from older revisions.

## Completion
Concisely report the change, actual commands and outcomes, source/data provenance, remaining limitations and any unverified claims. Do not silently unlock OOS, alter sealed evidence, or ask for approval for routine reversible work.