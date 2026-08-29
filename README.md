# Market Neural Net — Self-Learning Trading Agent for Indian Equities

A black-box neural agent that is given **no hand-engineered features, no indicators, no
trading rules**. It is given raw price/volume history for the Indian market, learns its own
representation of "what patterns exist across time", trades a simulated book, is scored on
realized risk-adjusted P&L, and keeps updating its weights as new data arrives.

---

## 0. Reality check (read this before writing code)

This plan is built to actually work, which means being honest about the hard parts:

1. **Signal-to-noise in daily equity returns is ~1–5%.** A model that fits 95% of the
   variance is fitting noise. Expect numbers that look like failure (52% hit rate) to be the
   *good* outcome.
2. **"Learns from its mistakes in live markets" is reinforcement learning, and RL is
   sample-starved here.** 25 years of daily bars is ~6,000 timesteps per stock. Atari agents
   need millions. This is why the plan spends most of its effort on *self-supervised
   pretraining* (which has millions of training targets) and uses RL only as a thin
   fine-tuning layer on top.
3. **Markets are non-stationary and adversarial.** A pattern that worked 2015–2020 can invert.
   The continual-learning design (Phase 6) is not a nice-to-have; it is the core of the project.
4. **The backtest will lie to you.** Lookahead bias, survivorship bias, and multiple-testing
   bias will each independently manufacture a fake Sharpe of 2+. Phase 4 exists entirely to
   stop that.
5. **Most such projects fail to beat buy-and-hold NIFTY after costs.** Phase 7 defines explicit
   kill criteria so we find out in month 3, not year 3.

The goal is a rigorously evaluated agent, not a money printer. Treat every result as guilty
until proven innocent.

---

## 1. Design principles

| Principle | Consequence |
|---|---|
| **No handcrafted alpha** | No RSI/MACD/Bollinger. Inputs are raw OHLCV + calendar structure only. The model must discover any such construct itself. |
| **Learn representation before learning to trade** | Self-supervised pretraining on all data, then a small trading head. Avoids burning scarce reward signal on learning basic market structure. |
| **Costs are part of the objective, not a post-hoc deduction** | STT, brokerage, stamp duty, slippage, impact all enter the reward function during training. |
| **Causality is enforced structurally** | Causal masks, causal normalization, purged/embargoed splits. No feature at time *t* may touch data from *t+1*. |
| **Champion/challenger, never blind promotion** | New weights go to shadow mode first, promoted only on out-of-sample criteria. |

**Scope decision (locked in): advisory before autonomous.** The end goal is still a fully
autonomous trading agent — that doesn't change. But the near-term deliverable is a
**recommendation engine**, not an auto-trader: for each candidate name, output what to buy,
at what entry price, target price, stop-loss price, and quantity/position size — something a
human reviews and acts on manually. Automation is a later, explicit graduation once the
recommendations have earned trust on their own track record, not a default. This reorders
Phase 7 (§4): an **advisory mode** (recommendations only, logged and scored, no order
placement) comes before paper trading with real order simulation, which comes before live
automated execution. See Phase 5/7 for what this changes concretely.

**Scope decision (locked in):** cash equities only, for now. Futures & options are a real
possibility later — the repo is laid out so an `data/raw/fno/` sibling and an options-specific
encoder slot in cleanly — but nothing in Phases 0–5 requires them, and F&O brings margin,
Greeks, and expiry-roll complexity that would slow down getting the core pipeline right. Kite
Connect (₹500/month) is **not** used yet either: every data source in Phase 1 is free
(NSE's own public bhavcopy archives + corporate-actions API — see §4). Kite only enters the
picture if/when Tier B minute-bar data is actually needed; until then there's no reason to pay
for it.

---

## 2. Tech stack

- **Python 3.11**, PyTorch 2.7
- **Data**: Polars + Parquet (partitioned by symbol/year), DuckDB for ad-hoc queries
- **Experiment tracking**: MLflow (local file store, syncable) or Weights & Biases (better
  when training moves between machines — see §2.1)
- **Config**: Hydra / OmegaConf YAML — every run reproducible from a config hash
- **Backtest**: custom vectorized engine (not vectorbt/backtrader — they hide assumptions we
  need to control)
- **Broker APIs (deferred, paid, only if Tier B is actually needed)**: Zerodha Kite Connect
  (₹500/mo, minute history from 2015) is the plan if/when minute bars matter; Angel One
  SmartAPI (free) and Dhan/Upstox/Fyers are fallbacks. Not used for Phase 1 — the daily-bar
  dataset comes entirely from NSE's own free public archives.

### 2.1 Split compute: local CPU + Colab / friend's GPU

This box is CPU-only (torch 2.7.0+cpu), so the workflow is explicitly two-machine. Design for
it from day one rather than retrofitting later.

**What runs where**

| Workload | Where | Why |
|---|---|---|
| Data ingestion, cleaning, corporate actions, universe build | **Local CPU** | I/O-bound, not compute-bound. Runs fine overnight. |
| Backtests, walk-forward evaluation, metrics | **Local CPU** | Vectorized numpy/polars; a full walk-forward on daily bars is seconds to minutes. |
| LSTM/TCN baselines on daily bars | **Local CPU** | Small enough (~1M params, 12M rows) to train in minutes–hours. Keeps iteration fast. |
| Transformer SSL pretraining | **Colab / friend's GPU** | 15–25M params × millions of samples. 20–100× speedup on a GPU. |
| Minute-bar pretraining | **Friend's GPU** (needs long uninterrupted runs) | Colab's session limits make multi-day runs painful. |
| RL / PPO fine-tuning | **Either** | Modest compute, but many rollouts — GPU preferred. |

**Rules that make two-machine training painless**

1. **Device-agnostic code.** One `DEVICE = torch.device("cuda" if torch.cuda.is_available()
   else "cpu")` in a single place. Never hardcode `.cuda()`. Every script must run on CPU with
   a tiny config for smoke-testing before it's shipped to the GPU.
2. **Checkpoint every N steps to durable storage, and always resume from checkpoint.** Colab
   disconnects — a run that can't resume is a run you'll lose. Save `{model, optimizer,
   scheduler, step, rng_state, config_hash}`. Target: killing the process at any moment costs
   at most 10 minutes of work.
3. **Ship data, not notebooks.** The curated parquet is the interface between machines. Daily
   Tier A is under 1 GB — put it in Google Drive and `drive.mount()` it, or upload to a GCS/S3
   bucket / HuggingFace dataset repo. Minute Tier B (15–30 GB) is too big for repeated Colab
   uploads: keep it on the friend's machine, or store it in a bucket and stream shards.
4. **Repo comes from git, not from copy-paste.** ✅ Done — the project is on GitHub at
   [github.com/yuvidewan/market_neural_net](https://github.com/yuvidewan/market_neural_net)
   (private). `notebooks/colab_train.ipynb`'s setup cell is `git clone`/`git pull`, so the
   notebook holds zero logic — it's a launcher, never source of truth. Data still doesn't go
   through git (large, regenerable, gitignored) — that part still goes via Drive per rule 3.
5. **Config-driven scale.** `configs/model/transformer_small.yaml` (CPU smoke test) and
   `transformer_full.yaml` (GPU) differ only in dims/steps, so there's one code path.
6. **Results flow back as artifacts.** GPU machine writes checkpoint + metrics JSON + run
   config to the shared bucket/Drive; local machine pulls them and runs the *evaluation*
   (which is CPU work anyway). This keeps the backtest harness — the part that must never be
   rushed — on the machine you control.
7. **Track runs remotely.** W&B is worth it here specifically because runs on three different
   machines land in one dashboard.

**Colab specifics**
- Free tier: T4 (16 GB), sessions cut off after a few hours and on idle. Enough for daily-bar
  transformer pretraining if checkpointing works.
- Colab Pro (~₹1,000/mo) gives longer sessions and better GPUs; worth it around M3–M4 if a
  friend's GPU isn't consistently available.
- Keep a `notebooks/colab_train.ipynb` that is: mount Drive → clone repo → pull data → run
  `python -m src.train --config-name=<x> --resume` → sync checkpoints back.

**Friend's GPU specifics**
- Prefer SSH + `tmux` + `rsync` over remote-desktop-and-click. A run should survive your
  laptop closing.
- Pin the environment (`requirements.txt` with exact versions, or a Dockerfile) so their CUDA
  version doesn't silently change results.
- Agree on a data location and never edit the curated dataset on two machines — one machine
  is the producer, everyone else pulls.
- **Paid GPU is the fallback, not a failure.** RunPod/Vast.ai A100 runs ~$1/hr; the whole
  project through M5 is roughly $100–300 of GPU time. Cheaper than losing weeks to a queue.

---

## 3. Repository layout

```
market_neural_net/
├── README.md
├── configs/                 # hydra yaml: data, model, train, backtest, live
├── data/
│   ├── raw/                 # untouched broker/exchange dumps
│   ├── interim/             # parsed, symbol-mapped
│   └── curated/             # adjusted, survivorship-clean parquet (the training set)
├── src/
│   ├── data/
│   │   ├── ingest/          # yfinance, bhavcopy, kite, smartapi loaders
│   │   ├── corporate_actions.py
│   │   ├── universe.py      # point-in-time index membership / liquidity filters
│   │   └── dataset.py       # torch Dataset, causal windowing, normalization
│   ├── models/
│   │   ├── encoders/        # lstm.py, tcn.py, transformer.py
│   │   ├── heads/           # quantile_return.py, vol.py, policy.py
│   │   └── ssl/             # masked modeling, autoregressive objectives
│   ├── rl/
│   │   ├── env.py           # portfolio simulator (gym-style)
│   │   ├── costs.py         # Indian cost model
│   │   └── ppo.py, dsr.py   # PPO + differentiable-Sharpe optimizer
│   ├── eval/
│   │   ├── walkforward.py   # purged CV with embargo
│   │   ├── metrics.py       # Sharpe, DSR, PBO, DD, turnover, capacity
│   │   └── baselines.py
│   ├── continual/           # replay buffer, EWC, regime detector, promotion gate
│   └── live/                # paper trading loop, risk layer, kill switch
├── notebooks/               # colab launchers + exploration only, never source of truth
├── scripts/                 # one-shot CLI entrypoints
└── tests/                   # lookahead tests are mandatory, see §4.4
```

---

## 4. Phased plan

### Phase 0 — Scaffolding (week 1)

- Repo, venv, config system, logging, experiment tracking, pytest, pre-commit.
- Decide the universe: **NSE cash equities**, point-in-time NIFTY 500 membership plus any stock
  with 60-day ADV > ₹5 crore. Include delisted/dropped names (survivorship).
- Set up the two-machine workflow from §2.1 now, while the code is small.
- Deliverable: `python -m scripts.hello_data` prints one clean adjusted price series, and the
  same command runs unchanged in Colab.

### Phase 1 — Data foundation (weeks 2–4) — *the phase that decides the project* — **in progress (M1)**

**Tier A — daily bars, full history, full universe (start here) — cash equities only**
- Source: NSE's own public bhavcopy archives at `nsearchives.nseindia.com`, confirmed live and
  working back to 1998 across both the pre-2024 format and the newer UDIFF format (automatic
  fallback between the two — see [`bhavcopy.py`](src/data/ingest/bhavcopy.py)). No login, no
  paid API, no yfinance dependency for this.
- Scale: ~2,500 symbols × ~6,500 sessions ≈ 12M rows ≈ under 1 GB parquet. Fits in RAM, fits
  in Drive, trains on CPU. Enough to build and validate the entire pipeline end to end.
- Series filter: **`EQ` only** (standard equity). `BE`/`BZ`/etc. and non-equity CM-segment
  instruments (T-bills, SGBs, GS) are parsed but excluded from the curated equity set.

**Tier B — 1-minute bars — deferred, not started.** Needs Kite Connect (₹500/mo) or an
equivalent paid feed, which per the current scope decision (§1) only gets bought if/when
minute-bar data actually becomes necessary. When it does: top 200 liquid names × 375
bars/day × ~2,700 days ≈ 200M rows ≈ 15–30 GB parquet, downloaded the same
resumable-checkpointed way as Tier A.

**Futures & options — out of scope for now** (§1). The repo layout leaves room for
`data/raw/fno/` alongside `data/raw/bhavcopy/` when that becomes relevant; nothing in the
current pipeline assumes equities-only in a way that would need rework.

**Non-negotiable data work**
1. **Corporate actions** — ✅ implemented in
   [`corporate_actions.py`](src/data/corporate_actions.py): fetches NSE's real
   corporate-actions API and parses Bonus / Face-Value-Split text into backward price
   adjustment factors (dividends and scheme-of-arrangement/demerger events are deliberately
   *not* auto-adjusted — different methodology, not a generic ratio). Both raw and adjusted
   OHLC are kept.
2. **Survivorship** — ✅ handled structurally: curated prices come from actual per-day
   exchange files, so a delisted stock's history simply stops appearing at its real last
   session (`tradable_range.parquet`) rather than being filtered by today's membership.
3. **Point-in-time universe** — ⚠️ partial. True historical index-membership snapshots aren't
   freely available anywhere found; [`universe.py`](src/data/universe.py) fetches today's
   NIFTY 50/100/200/500 lists (explicitly dated, not backdated) and offers a **liquidity-based
   universe** (trailing-ADV threshold, computed only from data on/before the as-of date) as the
   point-in-time-safe alternative.
4. **Symbol mapping** — ✅ ISIN-keyed. NSE bhavcopy didn't carry an ISIN column before ~2011;
   handled via a symbol→ISIN backfill with a flagged synthetic fallback ID for names that
   delisted before ISIN tracking began (documented in `build_curated.py`).
5. **Circuit/halt flags** — not yet implemented (low-volume/no-trade-day artifact detection
   exists via the quality report, but no explicit band-hit flag yet).
6. **Data quality gate** — ✅ implemented in
   [`quality_report.py`](src/data/quality_report.py): missing sessions, zero-volume days,
   price ≤ 0, OHLC inconsistencies, stale-price streaks, with a GREEN/RED gate.

**M1 status: done.** Full 1998-01-01 → 2026-08-27 backfill complete: 7,094 trading days, zero
download errors, 9,220,885 curated equity rows across 5,287 ISINs (survivorship-inclusive).
43,267 raw corporate-action events fetched; 1,197 Bonus/Split adjustments parsed and applied
across 841 ISINs (174 bonus/split-worded events couldn't be parsed and are queued in
`unparsed_actions.csv` rather than guessed at). 40/40 tests passing.

Quality gate: **RED, by design, and left that way on purpose** — 37 rows (0.0004% of the
dataset) fail OHLC consistency, and every one of them clusters in 1998–2002 with the identical
signature (`open=high=low=0.00` alongside a valid `close`) — a known artifact of NSE's
earliest electronic reporting on illiquid sessions, not a pipeline defect. The gate is
zero-tolerance on OHLC arithmetic by design, so it correctly refuses to call this GREEN; the
decision was to document the artifact (bounded to pre-2003, fully characterized, traceable via
`data/curated/quality_report/failing_isins.csv`) rather than loosen the gate or silently drop
those rows. Revisit if a future phase's modeling actually touches those specific rows.

### Phase 2 — Input representation (week 5)

The model gets **raw market state only**. Per bar, per symbol:

- `log(close_t / close_{t-1})`, and the same for open/high/low relative to previous close
- `log(volume_t)` de-meaned by a *causal* rolling median (252d)
- realized-vol-scaled versions of the above (returns ÷ causal 20d realized vol) — variance
  stabilization, not a signal
- calendar structure: day-of-week, day-of-month, month, sessions-to-expiry, minutes-since-open
  (Tier B), is-post-holiday
- a market-state token: NIFTY 50 return + India VIX, fed as a *separate token* rather than
  fused into the stock features — the model decides how to use it

Explicitly **not** included: any named indicator, any ratio anyone has published, any
fundamental, any sentiment. Those are exactly what the network should invent or ignore.

**Normalization must be causal**: rolling statistics from a trailing window only, never a
full-series `StandardScaler`. This single mistake is the #1 source of fake backtests.

### Phase 3 — Model progression (weeks 6–9)

Three encoders, in order, each a drop-in replacement behind one interface:

**v0 — LSTM baseline (must exist, must be beaten) — ✅ built, trained, M2 complete**
- 2-layer LSTM, hidden 128, sequence 120 daily bars, per symbol.
- Purpose: sanity floor. **Correction to an earlier assumption in this doc**: this is
  *not* fast local CPU iteration at real scale — 40 symbols × full 1998–2026 history × 4
  expanding walk-forward folds × 6 epochs took **6.5 hours on CPU** (see below). LSTMs are
  sequential over the time dimension, so there's no shortcut; use Colab for anything beyond a
  tiny smoke-test config once real folds/epochs are involved.
- **Real result (4 walk-forward folds, 2022–2025, 40 liquid ISINs, 10bps flat cost)**:
  gross Sharpe 0.47–0.56 and hit rate ~51% in every fold — consistently signed and
  consistently *slightly* above the shuffled-label null across 4 independent OOS years, which
  is a real if weak signal, not noise (the shuffled-label test already confirms this pipeline
  reports ~0 when there's nothing there). But **net Sharpe collapses to 0.12 overall**
  (one fold goes to ~0.00) once even a flat 10bps cost is applied, because the naive
  sign-of-prediction policy churns 45–73% of the book per day. That turnover collapse is the
  expected failure mode of the "just sign(prediction)" policy used here as a floor check —
  not a signal problem to fix in v0, but exactly what Phase 5's cost-aware, turnover-penalized
  policy exists to fix. v0's job (prove the harness works, beat the shuffled-label null) is
  done; making the signal itself better is v1/v2/Phase 5's job, not v0's.

**v1 — Dilated causal TCN — ✅ built and tested, M3 in progress**
- 8 blocks, dilations 1…128, kernel_size=2, 64 channels → receptive field exactly 256 bars
  (1 + sum(dilations)), causal structurally (left-padding only; tested directly by perturbing
  future timesteps and confirming past outputs are bit-for-bit unchanged).
- **"Still CPU-viable" needs a caveat, per the M2 lesson**: per-sample it's dramatically
  cheaper than the LSTM (convolutions parallelize across time; no sequential-recurrence
  bottleneck), but at M3's real target scale (200 symbols × full history × 4 walk-forward
  folds × 8 epochs) it's still GPU territory in practice — a single-epoch timing probe on 5
  symbols took ~1 minute, which projects to hours at full scale. **M3's real run goes to
  Colab from the outset** (`notebooks/colab_train.ipynb` §7) rather than repeating M2's
  overnight CPU run. Local runs are for smoke-testing correctness only (confirmed working).
- Self-supervised objective #2 (below) — quantile regression pinball loss — is implemented
  and encoder-agnostic (`src/models/ssl/quantile.py`), tested on synthetic data with
  hand-verified pinball-loss values and a real convergence check.
- Cross-sectional rank IC (`src/eval/metrics.py`) — the actual M3 gate metric — implemented
  and tested: grouped by date (not pooled across dates, which would confound the signal with
  day-level market moves), skips days with too few names for a meaningful correlation.

**v2 — Two-axis causal Transformer (the main model, trains on GPU)**
- Patchify: 16 consecutive bars → one token (PatchTST-style). 512 tokens ≈ long context.
- Alternating blocks: **temporal attention** (causal, RoPE, within one symbol) and
  **cross-sectional attention** (across all symbols at a fixed timestamp, unmasked).
  The cross-sectional axis is where sector rotation, lead-lag, and index effects get learned
  without anyone telling it what a sector is.
- d_model 256, 8 heads, 8 blocks ≈ 15–25M params. Small by LLM standards, right-sized for the
  data volume.

**Self-supervised objectives (the "learns whatever it can" stage)**

Trained on the *entire* dataset, all symbols, all history — millions of targets, no labels:
1. **Masked bar modeling** — mask 25% of patches, reconstruct them.
2. **Autoregressive next-bar prediction** — predict the *distribution* of the next return via
   quantile regression (pinball loss, 9 quantiles), not a point estimate. Markets are
   conditionally heteroskedastic; a point forecast discards the useful part.
3. **Auxiliary vol head** — predict forward realized volatility. Cheap, well-posed, and it
   forces the encoder to represent regime.

Multi-horizon return heads (1, 5, 20 bars) hang off the frozen encoder for evaluation. If the
encoder learned anything real, its embeddings carry predictive information — measure with
**rank information coefficient (IC)** out-of-sample *before* touching RL.
*Gate: mean OOS rank IC > 0.02 with stable sign, or go back to Phase 2.*

### Phase 4 — Evaluation harness (weeks 8–10, overlaps Phase 3)

Built *before* the trading agent, so no result is ever produced without it. Runs on local CPU.

**4.1 Splits** — walk-forward, expanding window. Train 2005–2015 → validate 2016 → test 2017;
roll forward one year at a time. **Purge** any training sample whose label horizon overlaps the
validation window, and **embargo** 10 sessions after it (López de Prado). No shuffled k-fold on
time series, ever.

**4.2 Portfolio simulator**
- Fill assumption: next bar's VWAP (never same-bar close), plus slippage.
- Slippage: `k · σ_t · sqrt(order_size / ADV)`, with `k` calibrated conservatively.
- Circuit-limit days: no fill.
- T+1 settlement, lot/tick rounding, ₹ position granularity.

**4.3 Indian cost model** (`src/rl/costs.py`), intraday and delivery variants: brokerage,
STT, exchange transaction charges, SEBI turnover fee, stamp duty, GST on (brokerage + txn
charges), DP charges on delivery sells. Round-trip cost lands roughly **12–35 bps** depending
on segment and broker — verify current rates against your broker's published schedule, they
change. A strategy at 200% monthly turnover needs ~5%+ annual gross alpha just to break even,
so the reward function must know this.

**4.4 Anti-fooling tests (in CI)**
- `test_no_lookahead` — blank all future data, assert outputs unchanged.
- `test_shuffled_labels` — train on shuffled targets, assert Sharpe ≈ 0. If it isn't, there's
  a leak.
- `test_cost_sensitivity` — report Sharpe at 1×, 2×, 3× modelled costs. If 2× kills it, it
  isn't real.
- **Deflated Sharpe Ratio** and **Probability of Backtest Overfitting**, reported on every run,
  accounting for how many configurations have been tried so far.

**4.5 Baselines that must be beaten out-of-sample, after costs**
buy-and-hold NIFTY 50 TRI · equal-weight universe · 12-1 cross-sectional momentum ·
50/200 MA crossover · random-weight portfolio at matched turnover.

### Phase 5 — Trading agent / RL (weeks 10–13)

Encoder pretrained and frozen, then unfrozen at low LR. A policy head is attached.

**Action space** — target portfolio weights over the universe: continuous `w_t` with
`Σ|w_i| ≤ L` (start `L = 1.0`, long-only; add shorts later), `|w_i| ≤ 5%`, sector exposure
`≤ 25%`. Actions are *target weights*, so turnover — and therefore cost — is an explicit,
differentiable function of the policy.

**Reward** (per step, log space):

```
r_t = w_{t-1}·ret_t − cost(|w_t − w_{t-1}|) − λ_dd·max(0, DD_t − DD_cap) − λ_turn·|Δw|
```

Optimized for **risk-adjusted** outcome, not raw return.

**Two-stage optimization** (ordering matters):
1. **Differentiable Sharpe first.** With the simple cost/fill model the simulator is
   differentiable, so directly maximize the *differential Sharpe ratio* by backprop. Orders of
   magnitude more sample-efficient than policy-gradient RL — it converges on 6,000 timesteps
   where PPO will not.
2. **PPO second, for the non-differentiable frictions** (circuit halts, discrete lots, impact,
   partial fills). Recurrent/transformer policy, GAE(λ), entropy bonus, gradient clipping.
   Small LR — this is fine-tuning, not learning from scratch.

**Guard against degenerate policies**: an agent that discovers "hold NIFTY forever" is
technically correct and useless. Always report *excess* return over buy-and-hold, and penalize
market beta in the reward when the goal is genuine alpha.

**Advisory output layer (§1 scope decision).** The policy's target weights are an internal
representation, not what a human sees. A separate translation step turns `w_t` (and the
model's own predictive distribution, since the quantile-regression SSL head already produces
one — see Phase 3) into a per-name recommendation card:
- **Entry price** — last close, or a limit price if the model's edge is conditional on getting
  filled at a specific level (derived from the predicted return distribution, not just its mean).
- **Target price** — set from the predicted return distribution's upper quantile over the
  policy's intended holding horizon, not an arbitrary round-number multiple.
- **Stop-loss price** — set from realized/predicted volatility (e.g. a multiple of predicted
  ATR or the lower quantile), never a flat "-5%" rule — the whole point of a learned model is
  that risk isn't the same across names and regimes.
- **Quantity** — the position size the policy's `w_i` implies at the user's actual capital and
  risk budget, converted to whole lots.
- **Confidence/rationale signal** — at minimum the model's own predicted-return quantile
  spread (tighter = more confident); full natural-language rationale is a nice-to-have, not
  required for the advisory mode to be useful.

This is a reporting layer on top of the same policy that will eventually place orders
autonomously — building it doesn't add new modeling work, it adds a translation from `w_t`
to something a human can act on, plus the tracking needed to score whether those
recommendations were actually good (which becomes the trust record that later justifies
turning automation on).

### Phase 6 — Continual learning (weeks 13–16) — *"improves with time"*

The mechanism behind the requirement that it keeps learning on its own.

1. **Scheduled walk-forward retraining** — monthly full retrain on the expanded window; weekly
   low-LR fine-tune on recent data.
2. **Catastrophic-forgetting defense** — experience replay buffer stratified by *regime*
   (2008 crash, 2013 taper, 2020 covid, 2021 bull, 2022 rate shock, 2024–25). Each update mixes
   ~70% recent / 30% replay. Add **EWC** or **L2-SP** regularization toward the previous
   champion's weights so drift stays gradual.
3. **Regime awareness, learned not labelled** — cluster the encoder's own embeddings
   (k-means / HMM on the latent state), then optionally a **mixture-of-experts** policy gated by
   latent regime, so the agent holds multiple behaviours instead of averaging them into mush.
4. **Champion/challenger promotion gate** — a newly trained model becomes *challenger* and
   trades in shadow for 20 sessions. Promotion requires: OOS Sharpe ≥ champion, max DD not
   worse by >20%, turnover within band, and no single position driving >40% of P&L. Otherwise
   the champion stays. **Never auto-promote on training loss.**
5. **Drift detection** — monitor rank-IC decay and feature-distribution shift (PSI). An alarm
   triggers retraining or de-risking, never silent continuation.

### Phase 7 — Advisory mode, then paper trading, then (maybe) live (weeks 16+)

**7a. Advisory mode (comes first — this is the actual near-term goal, not a formality).**
Every trading day, the current champion model runs inference on the live/latest data and emits
recommendation cards (§ Phase 5's advisory output layer) — no orders placed anywhere, nothing
automated. Logged to a simple daily report: symbol, entry, target, stop-loss, quantity, and the
confidence signal. Critically, every recommendation is **scored against what actually
happened** (did price reach target / stop-loss / neither, and when) so there's a real, growing
track record before any automation conversation happens. This is the stage where you're
looking at what it suggests and deciding whether you'd have taken the trade.

**7b. Paper trading (order simulation, still no real money).**
1. **Paper trade for at least 3 months** — broker sandbox, live feed, real timestamps. Log every
   intended order together with the model state that produced it.
2. **Reconcile paper vs backtest.** If live paper P&L diverges materially from what the backtest
   claims for the same period, the backtest is wrong. Fix it before risking anything.

**7c. Automation (opt-in graduation, not a default).**
3. **Risk layer sits outside the network and can always override it** — per-symbol cap,
   gross/net exposure cap, daily loss limit, max-drawdown kill switch, order-rate limiter,
   fat-finger price bands, and a "flatten everything" button.
4. **Regulatory** — SEBI has tightened rules on retail algorithmic trading through broker APIs
   (algo registration/tagging with broker and exchange, order-rate thresholds). Confirm the
   current requirements with your broker before any automated live order flow, and route through
   their approved algo path.
5. **If it goes live** — start at an amount you'd be fine losing entirely. Scale only on realized
   out-of-sample track record, never on backtest confidence. Even once automated, keep the
   advisory report running independently as the audit trail of what the model intended.

---

## 5. Milestones and gates

| # | Milestone | Gate to pass before proceeding |
|---|---|---|
| M1 | Curated daily dataset, full universe, survivorship-clean | Data-quality report green; lookahead tests pass |
| M2 | ✅ LSTM baseline trains; walk-forward harness runs end to end | Shuffled-label test gives Sharpe ≈ 0 — passing |
| M3 | Self-supervised encoder pretrained (first GPU run) — code done, real run pending Colab | OOS rank IC > 0.02, stable sign across folds |
| M4 | Differentiable-Sharpe agent backtested | Beats all 5 baselines OOS after costs; survives 2× costs |
| M5 | PPO fine-tune + continual-learning loop | Rolling 12m Sharpe stable across ≥3 walk-forward folds |
| M6 | Advisory mode: daily recommendation cards, scored | ≥60 scored recommendations with a real (even if modest) hit-rate edge over random |
| M7 | 3 months paper trading | Paper vs backtest divergence within tolerance |

**Kill criteria — stop and rethink rather than tune, if:** M3 fails after three genuine
architecture attempts; or the agent cannot beat 12-1 momentum after costs; or Sharpe collapses
under 2× cost sensitivity; or PBO > 0.5.

---

## 6. Metrics reported on every run

CAGR · annualized volatility · **Sharpe** · Sortino · Calmar · max drawdown & duration ·
**Deflated Sharpe Ratio** · **PBO** · rank IC and IC-IR · hit rate · profit factor ·
average win / average loss · annual turnover · average holding period · gross & net exposure ·
beta to NIFTY · **alpha net of beta** · capacity estimate (₹ at which slippage eats 50% of the
edge) · worst 10 days · per-year and per-regime breakdown.

Single-number summaries are banned. Every result carries its cost assumption and its fold count.

---

## 7. Compute and cost budget

| Item | Where | Estimate |
|---|---|---|
| Data ingestion + curation | Local CPU | Overnight runs, free |
| Daily-bar LSTM/TCN experiments | Local CPU | Minutes to a couple of hours per run, free |
| Transformer SSL pretraining, daily bars | Colab T4 / friend's GPU | ~4–10 GPU-hours (free tier is enough with checkpointing) |
| Transformer pretraining, minute bars | Friend's GPU or rented A100 | ~50–150 GPU-hours (~$50–150 if rented) |
| RL fine-tuning sweeps | Colab / friend's GPU | ~10–30 GPU-hours per full sweep |
| Backtesting + evaluation | Local CPU | Free — and deliberately kept local |
| Kite Connect data + API | — | ₹500/month, **deferred** — only bought if/when Tier B (minute bars) is actually needed; Tier A daily equities is 100% free |
| Colab Pro (optional, from M3) | — | ~₹1,000/month |
| **Total to reach M5** | | **~$0–300 of compute + a few months of work** |

---

## 8. Immediate next steps

1. `scripts/setup_env.py` — venv, pinned deps, directory tree, tracking store.
2. `src/data/ingest/bhavcopy.py` — resumable NSE bhavcopy downloader, 2000 → today.
3. `src/data/corporate_actions.py` + validation report.
4. `src/data/universe.py` — point-in-time membership + liquidity filter, delisted names retained.
5. `tests/test_no_lookahead.py` — written *before* the first model.
6. `src/models/encoders/lstm.py` + walk-forward harness → get M2 green on local CPU.
7. `notebooks/colab_train.ipynb` — the launcher shell (clone → data → resume-train → sync back),
   verified with the tiny config before any real GPU run.

Build in that order. The temptation is to start with the model; the projects that fail are the
ones that did.
