# ONEFILE

Single-file quantitative trading research & event study engine for **4H Crypto Perpetuals** (`BTCUSDT` USD-M).

Built around a strict **Hypothesis → Experiment → Decision (`PASS / KILL`)** methodology: no indicator soup, no silent forward-filling, no lookahead leakage, and mandatory month-matched control benchmarks.

---

## The 8-Stage Research Roadmap

| # | Stage | Output | Status |
| :-: | :--- | :--- | :--- |
| **01** | **Research Universe** | Assets (`BTCUSDT`), `4h` timeframe, long/short scope, `42`-bar (7d) horizon | ✅ Completed |
| **02** | **Market Behavior** | Trend, momentum, volatility, liquidity & regime observations | 🔄 In Progress |
| **03** | **Economic Mechanism & Edge Hypothesis** | Single falsifiable mechanism explaining *why* the setup has positive expectancy | ✅ Brick 1 Defined |
| **04** | **Mathematical Formalization** | Exact entry, stop, target, time-stop & invalidation rules | ✅ Brick 1 Locked |
| **05** | **Realistic Backtest** | Taker fees, slippage, historical 8h funding rates, conservative `STOP-FIRST` tie-breaking | ✅ Implemented |
| **06** | **Scientific Validation** | Bootstrap 95% CI, permutation $p$-value vs. control, yearly regime breakdown, locked OOS (`2025+`) | ✅ Implemented |
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
* **Execution & Cost Model ([`evaluate`](one.py)):**
  * Stop: `Entry - 1.5 * ATR(14)` (`STOP-FIRST` on ambiguous candles, gap-down fills at open).
  * Target: `+3.0R` (`--target-r`), Time Stop: `42` bars (`7d`).
  * Friction: `4.0 bps/side` fee + `2.0 bps/side` slippage + historical 8h Binance USD-M `fundingRate`.

### DEV Results (`BTCUSDT` 4H, `2020-01-01` – `2024-12-31`, `10,962` Candles)

| Group | Entries ($N$) | Win Rate | Mean Net $R$ (ex-fnd) | Mean Net $R$ (w/ fnd) | 7d MFE (`P50 / P75 / P90`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Setup (`3.0R` Target)** | `66` | **37.9%** | `+0.568 R` | **`+0.524 R`** | `+2.69R / +4.94R / +9.61R` |
| **Month-Matched Trend Control** | `66` | 18.2% | `-0.029 R` | `-0.081 R` | `+2.11R / +3.75R / +5.81R` |

* **Bootstrap 95% CI (Setup Mean Net $R$):** `[+0.073R, +0.988R]` ($P(\mu > 0) = 98.9\%$)
* **Permutation Test (Setup > Control):** $\Delta = +0.606\text{ R}$, one-sided **$p = 0.0250$ (`PASS`)**

---

## Quickstart

```bash
# 1. Install dependencies (via uv)
uv sync

# 2. Run deterministic safety & causality self-tests
uv run python one.py --self-test

# 3. Download canonical Binance USD-M 4h candles + funding rates and run DEV study
uv run python one.py --download
```
