# ONEFILE — Speed First

ONEFILE is a single-file (`one.py`) crypto perpetual futures research project using vectorbt. Optimize for fast, correct task completion. Work autonomously, trust your engineering judgment, and keep the solution small.

## Working

- For a clear task, locate the relevant code and implement it directly. Skip formal plans, broad audits, and unrelated cleanup unless the task needs them.
- Read only what you need. Read all of `one.py` only when the change requires understanding the whole file. Reuse context; do not reread unchanged code or instructions.
- Batch independent searches and reads. Make coherent edits before checking them; avoid edit/check loops after every small change.
- Reuse existing functions, dependencies, data, and cached results. Add files, abstractions, dependencies, or tests only when they solve a concrete need. Keep research logic in `one.py`.
- Consult official docs when API behavior is uncertain; stop researching once the uncertainty is resolved. Do not ask the user about routine, reversible implementation choices.

## Verification

- Choose the smallest check that meaningfully covers the changed behavior. There is no mandatory full-suite run at task end.
- Documentation, comments, and formatting need no runtime tests. Local behavior changes normally need only the relevant existing test or a focused check. Add a regression test when it provides lasting value.
- Run the full suite for broad changes whose effects cannot be covered reliably with focused checks, or when explicitly requested. Run it once on the final revision; after a fix, rerun affected checks and broaden only if needed.
- Reuse passing results for unchanged behavior. Do not rerun checks merely for reassurance. Never call an older revision verified if later edits affect what was tested.
- Do not launch downloads, long experiments, or full research runs to validate an ordinary code edit. Reuse valid market and funding files. Retry a failed command only after addressing its cause or changing approach.
- Report exactly what ran and passed. Focused checks are not a full-suite pass; synthetic tests are not market evidence.

## Performance and research

- For speed tasks, time the requested operation, fix the measured bottleneck, and compare equivalent before/after runs. Separate cold startup/JIT from warm execution. Do not claim speedups without measurements or change results to meet a time target.
- Use DEV data only. Never access or unlock sealed OOS (`--unlock-oos`) without explicit user authorization.
- Preserve leakage controls, realistic execution, fees, slippage, and funding. Exploratory results remain hypotheses; stale evidence is not current proof.

Finish briefly: what changed, checks actually run, and any material uncertainty. Stop when the requested work is complete.
