# ONEFILE — Agent Instructions

ONEFILE is a single-file (`one.py`) research project for crypto perpetual futures, using vectorbt. Work autonomously. Trust your own engineering judgment: keep changes simple and focused, reuse existing code, and avoid unnecessary frameworks, files, or process.

- Read `one.py` fully when necessary to understand the task; afterwards, read only the relevant unchanged sections. Consult official docs when API behavior is uncertain.
- Keep research on DEV data. Never access or unlock sealed OOS (`--unlock-oos`) without explicit user authorization.
- Avoid leakage, overfitting, and unrealistic trading assumptions. Account for fees, slippage, and funding. Treat exploratory results as hypotheses, not proven edge; never present stale evidence as current proof.

## Testing and iteration

- **Do not run the full test suite after each edit.** During development, run small, relevant checks only when useful. Use your judgment; no mandatory test after every change.
- **Run the full relevant test suite once at the end**, after the final changes, before claiming the work is verified. If it fails and you fix the code, verify the new final revision again. Never present an interrupted or outdated run as passing.
- Reuse existing market data and funding files when valid. Do not re-download, revalidate, or rerun long experiments without a concrete reason. Avoid repeating identical failing commands without a change in approach.
- Distinguish coding tests (including synthetic `--self-test`) from real-market research evidence. Preserve reproducibility and report actual results, not assumptions.

Finish with a concise summary of what changed, what was actually tested, and what remains unverified.
