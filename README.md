# ONEFILE

Single-file quantitative trading research & event study engine for **4H Crypto Perpetuals** (`BTCUSDT` USD-M).

Single-file **Hypothesis → Experiment → Decision** research workflow with fail-closed data checks and outcome-blind, month-matched control benchmarks. Outputs distinguish insufficient evidence, failed exploratory screens and exploratory passes; none proves a durable trading edge.

---

## The 8-Stage Research Roadmap

| # | Stage | Output | Status |
| :-: | :--- | :--- | :--- |
| **01** | **Research Universe** | Assets (`BTCUSDT`), `4h` timeframe, long/short scope, `42`-bar (7d) horizon | ✅ Completed |
| **02** | **Market Behavior** | Trend, momentum, volatility, liquidity & regime observations | 🔄 In Progress |
| **03** | **Economic Mechanism & Edge Hypothesis** | Single falsifiable mechanism explaining *why* the setup has positive expectancy | ✅ Brick 1 Defined |
| **04** | **Mathematical Formalization** | Exact entry, stop, target, time-stop & invalidation rules | ✅ Brick 1 Locked |
| **05** | **Event Execution Study** | Native vectorbt fills, fees, slippage, external funding estimate, conservative `STOP-FIRST`; not a funded account | ✅ Implemented |
| **06** | **Scientific Validation** | Paired calendar-block sensitivity, declared-trial correction, yearly breakdown; independent validation/OOS (`2025+`) still required | 🔄 Exploratory only |
| **07** | **Risk Management & Paper Trading** | Position sizing, drawdown limits, order validation | ⏳ Planned |
| **08** | **Live Execution & Controlled Scaling** | Small capital deployment, monitoring, gradual scaling | ⏳ Planned |

---

## Brick 1: Price + Volume Momentum-Continuation (`4H`)

* **Core Question:** *After a strong volume-backed impulse inside a structural uptrend, when does a shallow pullback offer asymmetric continuation (`3.0R+`) superior to random trend entries in the same month?*
* **Mechanism ([`features`](one.py)):**
  1. **Trend Regime:** `EMA(20) > EMA(50)` and `Close > EMA(50)`.
  2. **Volume-Backed Impulse:** Bullish expansion bar (`Close - Open >= 1.0 * ATR(14)`) with `Volume >= 1.2 * SMA(Volume, 20)` within the prior 6 bars (`24h`).
  3. **Controlled Pullback:** Prior 1–3 bars pull back near `EMA(20)` (`<= 1.008 * EMA20`) while holding above `EMA(50)`.
  4. **Continuation Trigger:** Bullish close above `High[t-1]` and `EMA(20)` → entry at next bar `Open[t+1]`.
* **Execution & Cost Model ([`evaluate_events`](one.py)):**
  * Stop: `Entry - 1.5 * ATR(14)` (`STOP-FIRST` on ambiguous candles, gap-down fills at open).
  * Target: `+3.0R` (`--target-r`), Time Stop: `42` bars (`7d`).
  * Friction: `4.0 bps/side` fee + `2.0 bps/side` slippage + historical 8h Binance USD-M `fundingRate`.

### DEV Results (`BTCUSDT` 4H, `2020-01-01` – `2024-12-31`, `10,962` Candles)

Reproduced locally on 2026-10-17 using a **fresh Binance download**, then the existing CSV's entire funding history was compared with fresh canonical funding archives. All `5,481` settlements and `10,962` candles are present. Results are exploratory, **not account returns or independently validated edge**.

| Group | Entries | Mean Net R (funding included) |
| :--- | ---: | ---: |
| All setups, descriptive only | 66 | +0.528 R |
| Outcome-blind matched setup subset | 37 | +0.098 R |
| Disjoint same-month trend controls | 37 | -0.717 R |

**Current gate: `SCREEN FAIL`.** The matched setup mean does not exceed `+0.15R`; its calendar-block 95% CI envelope is `[-0.454R, +0.680R]`. Matched delta is `+0.815R` with CI envelope `[+0.093R, +1.487R]`. The largest centered-null bootstrap p across predeclared 1/3/6-month blocks is `0.0094`, **not a multiplicity-adjusted p or proof of edge**. The actual total number of prior trials is unknown by default, so no statistical PASS is available.

The 29 unmatched setups remain in the exported CSV (`matched=false`) and full-study summary. Controls are disjoint from **all 66 setup horizons**, not merely the matched subset. Partial common support changes the comparison population; the matched result cannot be generalized to all setups. No setups are selected or dropped using their returns. The old overlapping-control `p=0.0250` result is superseded. The real OOS holdout was not opened.

---


## E2E performance audit

Every CLI invocation (including failed studies and \`--self-test\`) writes an atomic
\`perf/<utc>-<pid>-<nonce>.json\` report with elapsed wall/CPU time, input SHA,
study mode, stage timers, vectorbt kernel vs adapter timings, and an AST
inventory of every Python \`for\`/\`while\`/comprehension in \`one.py\`.
Scientific evidence and sealed OOS remain separate.

For a sampled call tree and interactive HTML flamegraph, optionally install
[Pyinstrument](https://github.com/joerick/pyinstrument) **in the active environment**:

\`\`\`bash
uv pip install --python .venv/bin/python pyinstrument
uv run --no-sync python one.py --csv core5_4h.csv --offline --multi-research --protocol core5_perf_attempt01.json
\`\`\`

Reports are in \`perf/\`. Without Pyinstrument, E2E/timers/loop inventory
still save automatically; \`sampling\` records why flamegraph output is absent.
No new dependency is required for normal execution and \`uv sync --frozen\`
still works. \`uv sync\` may remove ad-hoc profiler packages; reinstall as needed.

For **exact Python line hit counts** (much slower, not for performance comparisons),
use [line_profiler](https://github.com/pyutils/line_profiler):

\`\`\`bash
uv pip install --python .venv/bin/python line-profiler
uv run --no-sync python one.py --csv core5_4h.csv --offline --multi-research --protocol core5_perf_attempt02.json --profile-loops
\`\`\`

The \`.lprof\` file and per-loop line-hit estimates accompany the JSON audit.
Python source loops that were never entered show zero hits only in deep mode.
JIT-compiled Numba loop iterations cannot be counted by Python line tracing.
Loop-body line times are approximate and nested loops overlap.

Compare two runs **with the same input, study mode and profiler mode**:

\`\`\`bash
uv run --no-sync python one.py --offline --csv btc_4h.csv --perf-baseline perf/OLD.json
\`\`\`

The next JSON contains measured \`wall_reduction_percent\` and \`speedup_x\`.
For cold/warm JIT comparisons, run at least two identical invocations; clearly
distinguish process startup/JIT from actual simulation. Use \`--no-perf\` for
unprofiled timing controls. Never treat these timing reports as trading evidence.

## Quickstart

```bash
# 1. Install locked dependencies (Python 3.12 is the CI-tested runtime)
uv sync --frozen --python 3.12

# 2. Run deterministic execution tests and offline audit regressions
uv run python one.py --self-test
uv run python -m unittest discover -s tests -v

# 3. Download canonical Binance USD-M 4h candles + funding rates and run DEV study
uv run python one.py --download

# Existing CSV: verify ALL funding values against fresh Binance source (default)
uv run python one.py

# Explicit offline exploration: coverage still checked, source UNVERIFIED, no statistical PASS
uv run python one.py --offline --behavior

# Only if the complete number of previously inspected hypotheses/parameter variants is known:
# uv run python one.py --trials <actual_total_trials>
```

## Data safety & compatibility

The accepted CSV schema is now:

```text
time,open,high,low,close,volume,funding_rate,funding_observed,symbol
```

- `symbol` must be `BTCUSDT`; other CLI symbols are rejected. Multi-asset funding calendars are not implemented.
- `funding_observed` is `1` only when an actual settlement record was present in the downloaded archive, otherwise `0`. Every BTC 00:00/08:00/16:00 UTC settlement must be observed. A real zero rate is valid; an absent or non-finite settlement rate is not silently replaced by zero. Funding outside that schedule is rejected.
- **Legacy seven-column CSVs are deliberately rejected.** Back up the old file and run `uv run python one.py --download`. Do not manufacture observation flags for old zero-filled data; missing funding cannot be reconstructed from those zeros. The CSV fields are provenance declarations, not cryptographic proof against a deliberately forged input.
- Default online runs compare every funding settlement against canonical Binance monthly archives. ZIPs are downloaded once and cached under `~/.cache/onefile/binance` (override with `ONEFILE_CACHE_DIR`); each cached archive is SHA-256 checked before reuse. A mismatch or missing source record stops the run. `--offline` bypasses this comparison and marks source provenance UNVERIFIED; structural coverage alone is not proof of authenticity.
- Incomplete windows or failed data checks stop with a nonzero exit code **before** features, simulations or successful exports. The decision gate requires both matched groups to have at least 30 events, identical month counts, disjoint horizons and at least 12 active matched months. Insufficient controls do **not** crash the study: all setups are retained and the comparison uses only outcome-blind common support, or reports insufficient data.
- Statistics jointly resample monthly setup/control sums and counts, retaining empty months and local dependence under an approximate stationary-block model. Results include 1/3/6-month sensitivity and the worst p / widest CI envelope. `--trials` applies Bonferroni correction using the user-declared total history of hypotheses/variants, not just the current invocation. Unknown trial history never yields statistical PASS; even an exploratory pass requires independent validation.
- `--behavior` uses the same absolute UTC 43-bar grid for every condition, with all 43 phase offsets reported as sensitivity. Overlapping phases and conditions are descriptive, not independent samples or evidence for picking the best condition. H1-A future-confirmed onsets are retrospective labels, not tradable signals.
- Execution remains fixed 1 BTC with unlimited research cash. There is no $500-account sizing, margin, liquidation or account-equity claim. A real futures account engine remains outside this event study; do not convert R means to account returns.
- OOS input must include uninterrupted history from `2020-01-01` through the OOS end. Features use that past history; only OOS timestamps are eligible as OOS events. OOS remains opt-in with `--unlock-oos`.
- Input, trade-output and evidence paths must be distinct, including symlink/hardlink aliases. Downloads are fully validated before replacement; each file is written via an atomic temporary-file replacement. The two result files are not a multi-file transaction.

See [AUDIT_ONE_PY.md](AUDIT_ONE_PY.md) for the original findings and fix verification. The offline regression suite uses only synthetic data and local ZIP responses; it makes no exchange requests.

## Market understanding + H1-A feasibility (actual DEV run)

**Full Turkish evidence report: [RESEARCH_REPORT.md](RESEARCH_REPORT.md).** No confirmed economic edge was found. The existing Brick 1 result reproduced unchanged.

```bash
uv run python one.py --research
# Missing data: uv run python one.py --download --research
```

This separate mode retains the one-file implementation and existing vectorbt engine. It exports daily 1/3/7-day directional distributions, cost-after 3R/5R studies and a genuine **pre-onset** H1-A forecast experiment (M0/M1/M2, nested chronological selection/calibration, 17-day label purge, 28/56/84-day calendar-block uncertainty). Gross target hits and realized net-target attainment are different outputs. Funding is descriptive here, not a secretly added H1-A M3 feature.

Actual BTC 2020–2024 feasibility result: **346 outer-fold forecasts (340 mature/scored, 6 censored); only 5 up / 1 down confirmed onsets**, versus the 20/20 minimum. M2 Brier `0.210309` vs M1 `0.222566`; relative gain **5.51%**, CI **[-2.67%, +15.18%]**. M2 detected 2/6 events, with 2 false alarms and median genuine lead **1.5 days**, below the 2-day requirement. Verdict: **INCONCLUSIVE / low event power**, not success and not universal impossibility. No H1-A economic simulation was authorized by these results. Every descriptive 3R/5R mean CI included zero.

The source/data/protocol identity is frozen in `research_protocol.json` before computation. A mismatch fails closed; preserve prior attempts rather than deleting them to conceal revisions. This is a local historical feasibility protocol, **not external preregistration or independent OOS**. The first attempt and post-run schema/description revision are explicitly retained and reconciled in `research_integrity_audit.json`; the final source is archived. BTC-only scope, four outer folds and 4H-to-1D aggregation depart from the monograph's preferred core-5 setup. Both the existing **2025+ ONEFILE holdout** and **G3 >=2026-04-01** remain unopened; `--research --unlock-oos` is rejected before download.

Outputs: `research_evidence.json`, `market_observations.csv`, `h1a_forecasts.csv`, `h1a_onsets.csv`, `research_run.log`. Existing `*.csv` ignore rules remain; the evidence bundle preserves research CSV exports. Verification: **115 unit tests + 38 vectorbt self-tests + 24 actual-data future-corruption checks** passed.
