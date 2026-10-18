# ONEFILE development latency — measured 2026-10-18

Routine feedback now fits the <=5s target on the measurement host. The complete suite is still about 10s; neither all coding tasks nor this user's VPS have been timed. No relevant tests, bootstrap samples, folds, costs, or causality checks were removed.

## Changes

- Install the official precompiled `vectorbt[rust]` extra. The lock keeps vectorbt at **1.1.2**, adds matching `vectorbt-rust==1.1.2`, and leaves the installed versions of other packages unchanged. vectorbt's default `auto` dispatch uses Rust on supported paths and Numba otherwise. No custom backtester or global JIT-disable switch.
- Import vectorbt only when execution/engine metadata is used. Data checks, failed CLI validation, forecast-only tests and `--help` no longer initialize plotting/Numba dependencies.
- Select alarm threshold candidates with arrays and reuse event columns instead of filtering a DataFrame for every row. Keep probability ties, array thresholds, input event order, strict pre-onset matching, inclusive 7-day endpoint, cooldown and onset deduplication unchanged.
- Remove Pi edit/debounce/settlement verification hooks. `/onefile-verify` remains an optional manual command that executes the self-test once and rejects a stale or failed result. It no longer starts a separate redundant compile process.
- Include the existing execution self-test in unittest discovery. CI runs the Python suite in one interpreter. `AGENTS.md` directs ordinary work to one relevant check after the edits, with no obligatory baseline/full-suite cycle. Broad changes still need the full suite.

Python support is now **3.11–3.14**, required by the current vectorbt Rust release. CI continues to use Python 3.12. Deploy with `uv sync --frozen --python 3.12` once and reload/restart Pi so the old loaded hooks are removed; the VPS was not modified here.

## Measurements

Wall time includes `uv` and interpreter startup, captured using `time.perf_counter()` around subprocesses. Runs were sequential, without a profiler or other benchmark running concurrently. Each group started with an empty temporary `NUMBA_CACHE_DIR`, reused for the next two invocations. OS page caches were not purged. Installation and network downloads are excluded.

Host: `Linux-6.18.55-x86_64-with-glibc2.39`. Python **3.12.14**, vectorbt **1.1.2**, Rust extension **1.1.2**, NumPy **2.5.4**, pandas **3.0.6**, scikit-learn **1.9.1**.

Baseline source: `9eb6cb8fc77ee74f368414990d5d5b09626b15d4`.
Measured final `one.py` SHA-256: `075343adc7717edb08cb362089d1e4f60a91a853fa4d2a7c63b436b3e198d4d2`.

| Final check | Three wall times (seconds) | Median (seconds) | Coverage |
| --- | --- | --- | --- |
| Execution self-test | 2.4273, 2.5218, 2.8892 | 2.5218 | 38 checks |
| FileSafetyTests | 0.4461, 0.3396, 0.3533 | 0.3533 | 10 tests |
| ForecastTests | 3.0687, 3.1689, 3.1071 | 3.1071 | 12 tests |
| Full Python suite | 10.4824, 9.7124, 9.9248 | 9.9248 | 128 tests, including all 38 self-checks |

All runs exited **0**, with no skips. Extension tests: **7 passed**, **0.2062s wall**.

Baseline measurements on this same host:

- Original 38-check self-test with cold Numba cache: **16.9427s**, no profiler. With Rust installed for the experiment, the original module was run with `vbt.settings['engine']='numba'` to reproduce the original engine. Final self-test median **2.5218s**: **6.72x faster / 85.1% less wall time**.
- Original unittest discovery: **126 tests passed**, **26.279s wall**, **22.957s unittest time**. This ran before adding the Rust dependency, after the original self-test had populated its normal cache. Final discovery contains two additional tests (the 38-check self-test wrapper and alarm boundary regression) and has median **9.9248s**: **2.65x faster / 62.2% less wall time**. This is not a cold-cache comparison.
- An earlier **profiled** cold self-test took **29.661s** wall: **20.609s** cumulative in Numba compilation. These profiled numbers locate the bottleneck and are not used for the speedup ratio.
- Full-suite profiling after adding Rust identified `alarm_metrics`: **284 calls / 10.863s cumulative** under cProfile. Profiler overhead is excluded from the final table.

The user's reported 54s was not reproduced on the VPS. The table gives local measurements, not a guarantee for another machine. End-to-end model inference/coding latency was not measured. The full suite remains above 5s; ordinary changes should use the relevant class/method rather than rerun everything.

## Correctness evidence

- Existing 126 Python tests retained; final suite **128 passed**, including the same **38 execution assertions**. A new boundary regression tests probability/threshold ties, non-default row indices, unsorted events, repeated onset IDs, cooldown, signed matching and array thresholds.
- Compared the optimized alarm function with the original source on **90 deterministic reference cases** (seed 5342): 0/1/8/100/400 rows, empty/nonempty events, scalar/vector/NaN thresholds, confirmed/censored status and both signs. Complete result dictionaries matched exactly.
- Explicit `engine='numba'` vs `engine='rust'`: **20 batch comparisons / 280 event records per engine**, long/short, 7/42-bar holding, 3R/5R targets and barriers on/off, for both event adapters. All columns matched at **rtol=atol=1e-12**. Both engines separately passed the original **38 self-checks**, including tie handling, adverse/favorable gaps, fees, slippage, funding timing and causality.
- **7** Node extension tests passed, including no automatic hooks, exactly one manual execution, failure handling and stale-revision rejection. Actual Pi 1.3.0 hook loading on the VPS remains unverified.
- `uv lock --check` and `git diff --check` passed. No real-market or sealed OOS run was performed; committed historical research artifacts were left untouched.

## Reproduce current checks

```bash
uv sync --frozen --python 3.12
# Choose ONE relevant check for ordinary work; examples:
time uv run --no-sync python -m unittest discover -s tests -k FileSafetyTests
time uv run --no-sync python -m unittest discover -s tests -k ForecastTests
time uv run --no-sync python one.py --self-test
# Broad changes / CI: includes the self-test, so do not run it separately again.
time uv run --no-sync python -m unittest discover -s tests --durations 10
node --experimental-strip-types --test .pi/tests/onefile-extensions.test.mjs
```

## Primary sources consulted

- [vectorbt: optional precompiled Rust engine and installation](https://github.com/polakowo/vectorbt)
- [vectorbt: auto dispatch, supported paths and per-call engine selection](https://vectorbt.dev/getting-started/features/)
- [Numba: on-disk compilation cache](https://numba.readthedocs.io/en/stable/developer/caching.html)
- [Python unittest: test selection and duration reporting](https://docs.python.org/3.12/library/unittest.html)
