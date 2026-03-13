# MarketSim v3.0 — Market Making Simulator (C++ & Streamlit)

A market making sandbox built to learn the full loop end-to-end:  
**limit order book → external flow → MM quoting policy → fills → accounting → telemetry / trade tape → diagnostics.**

This **isn't** a high-fidelity exchange emulator. It's an **explainable simulation** designed to stress the right failure modes — inventory risk, adverse selection, regime bursts — while producing clean outputs you can sanity-check. Each architectural decision has a documented reason, not just a result.

---

## What changed from v2.x

v3.0 is a targeted response to four specific criticisms of the v2.2 architecture.

**[1] Cache-hostile memory layout → OrderPool + intrusive lists**  
`std::list<Order>` heap-allocates one node per order. That's 100–200 ns of allocator overhead per insert/erase, and because the nodes scatter across the heap, iterating a price queue during matching is a sequence of cache misses. v3.0 pre-allocates all `Order` objects in a single flat `std::vector` (the pool). Each `Order` embeds `pool_prev`/`pool_next` as `uint32_t` indices — intrusive doubly-linked lists that never touch the heap on the hot path. Alloc and free are free-list stack push/pop. Queue iteration is sequential through the flat array and actually benefits from the prefetcher.

**[2] O(log N) level lookup → ArrayBook with O(1) slot access**  
`std::map<price_t, Level>` costs O(log N) per lookup — fine in isolation, but that's 4–5 pointer hops per access for a realistic book depth, each a likely L2/L3 cache miss. `ArrayBook` uses a power-of-2 circular slot array (8192 ticks). The level for price P lives at `slots_[(P - base_) & kMask]` — a single array index. Best-bid/ask tracking uses `std::set<price_t>` (O(log k), but k is the number of live levels, typically < 30). The comments note what the production replacement would be: a Fenwick tree or cache-line-aligned bitset scan.

**[3] Constant latency → log-normal LatencyModel**  
Hardcoded `cancel_latency_s = 0.03` assumes every cancel arrives after exactly 30 ms. Real co-location RTT has a modal value near the hardware floor and a heavy right tail from kernel scheduling jitter, TCP retransmit, and exchange-side GC pauses. `LatencyModel` samples from `std::lognormal_distribution` with parameters calibrated to realistic co-location RTT ranges (mean ~30 ms, CV ~0.50 for cancels; mean ~10 ms for placements). An optional congestion penalty adds proportional delay when Hawkes intensity is elevated — modelling exchange queue backlog during flow bursts. The realised latency values are written to the trade tape so you can inspect the distribution.

**[4] OU fundamental as fair value → FairValueModel (microprice + imbalance momentum)**  
The Avellaneda-Stoikov derivation uses a noise-free fundamental price as its reference. A real MM does not have access to one. In v2.x the MM was quoting around the OU process — an unobservable latent variable. v3.0 replaces this with `FairValueModel`: the reservation price is derived entirely from observable book signals.

```
fair_value  = microprice + imb_coeff × imbalance_ema
reservation = fair_value − inv_skew × inventory
```

Microprice (Stoikov 2018) weights the mid by opposite-side queue size and is empirically validated as a short-term price predictor. The imbalance EMA captures sustained one-sided book pressure. The OU fundamental is retained only for the informed-trader model — it represents the latent value that informed agents observe but the MM cannot. That asymmetry is the point of the informed-flow component.

**[5] Generative-only → ReplayEngine for historical event replay**  
Hawkes processes are useful for studying dynamics in a controlled environment, but they cannot validate strategy behaviour against real order flow because they bake in the assumptions of whoever calibrated them. `ReplayEngine` ingests a normalised event CSV (`t_s, event, side, price_ticks, qty, order_id`) and drives the same `ArrayBook` and MM agent. `generate_synthetic()` produces self-contained test data in the identical format, so the replay pipeline runs without external files. To use real data: convert a Databento MBO or Binance L3 feed to the CSV schema, set `cfg.use_replay = true`. The MM strategy code is unchanged between modes — the only difference is the event source.

---

## What this project produces

Each run generates three files.

**Telemetry CSV** (`telemetry_v3_hawkes.csv` / `telemetry_v3_replay.csv`) — one row per step snapshot:
- state: time, BBO, MM quotes, inventory
- P&L: cash, realised/unrealised, equity
- microstructure: spread, imbalance, microprice, 5-level LOB depth each side
- signals: `fill_sign_ema` (toxicity), `tox_widen`, `fv_imb_ema` (imbalance momentum), `hawkes_intensity`

**Trades CSV** (`trades_v3_hawkes.csv` / `trades_v3_replay.csv`) — one row per fill:
- whether the MM was the maker, fill price, sign
- `maker_age_s`, `maker_queue_ahead_qty_entry`, `queue_depleted_before_fill` — adverse selection diagnostics
- `cancel_latency_s`, `place_latency_s` — realised latency samples for that requote cycle

**Multi-seed summary CSV** (`multi_seed_summary_v3.csv`) — one row per seed across 10 seeds:
- final equity, max drawdown, realised P&L, fees, fill count

---

## Architecture

### C++ engine (`market-sim-v3.0.cpp`)

**OrderPool**  
Pre-allocated slab of 65,536 `Order` objects in a flat vector. Each `Order` embeds `pool_prev`/`pool_next` as `uint32_t` indices. Alloc is free-list `pop_back`, free is `push_back`. No heap allocation on the matching hot path.

**ArrayBook**  
Circular slot array of 8,192 entries indexed by `(price_ticks - base_) & kMask`. Each `Slot` holds the head/tail of an intrusive queue plus `total_qty` (maintained incrementally — no O(n) scan) and `cumvol` (monotone cumulative traded volume for queue depletion tracking). Active price levels tracked in `std::set<price_t>` for O(1) best-bid/ask via `begin()`.

**Hawkes6**  
Six-dimensional Hawkes process (limit add bid/ask, cancel bid/ask, market buy/sell) with a full 6×6 asymmetric excitation matrix. Row sums all below 1.0 — process is stationary without relying on the intensity cap. Exact simulation via Ogata thinning, not Euler discretisation. Baseline intensities are state-dependent: spread, TOB imbalance, and information edge all modulate the baseline rates each event. `last_intensity()` exposes total λ for the congestion penalty.

**LatencyModel**  
`std::lognormal_distribution` parameterised separately for cancel and place latency. Optional congestion term adds `congestion_per_event_s × max(0, intensity − baseline)` seconds of extra delay during flow bursts. Disabled by default (`congestion_per_event_s = 0.0`).

**FairValueModel**  
`microprice = (ask × bid_qty + bid × ask_qty) / (bid_qty + ask_qty)`. `imb_ema` is an exponential moving average of `(bid_qty − ask_qty) / (bid_qty + ask_qty)`. Reservation price = `microprice + imb_coeff × imb_ema − inv_skew × inventory`. The MM has no access to `mark_` (the OU fundamental) in `plan_quotes`.

**MarketMaker**  
Toxicity-adaptive spread on top of the reservation price: `fill_sign_ema` tracks the EMA of fill direction (+1 = MM bought, -1 = MM sold). When `abs(fill_sign_ema)` exceeds a threshold, the spread widens by up to `tox_widen_max_ticks`. Cancel and place are scheduled as future `Action` objects on a min-heap keyed by `(t, seq)`, with separate log-normal latency samples per requote. `min_quote_life_s` prevents cancel spam.

**ReplayEngine**  
Reads or generates the historical event CSV. `generate_synthetic()` produces events in the same format as `load()` reads — so the same pipeline handles both test data and a real feed without any code change. The replay loop in `Simulator` is structurally identical to the Hawkes loop: same action queue, same MM agent, same `maybe_requote`. The strategy is agnostic to event source.

**Informed traders**  
22% of market orders are from informed agents (Glosten-Milgrom style) who only trade when the OU fundamental is far enough from the reference price. The MM cannot distinguish these from noise traders. This is what drives the adverse selection signal.

**Accounting identity**  
At every telemetry step: `equity = cash + inventory × mid_mark`. Checked in the Streamlit viewer as a residual plot. A non-zero residual indicates a bug in `apply_trade()`.

---

### Streamlit viewer (`market_sim_viewer_v3_0.py`)

Seven tabs.

**Overview** — equity curve with drawdown, TOB quotes vs OU fundamental (labelled as unobservable), inventory step chart, P&L components, rolling information ratio with an explicit disclaimer that it is not annualised to calendar time.

**Markout Analysis** — short-horizon markout at 1 / 5 / 10 / 30 simulated seconds. Convention: `markout_H = sign × (fill_px − mid(t + H))`. Positive = profitable from the maker's perspective. Summary table with mean, std, median, and percentage adverse at each horizon. Cumulative markout over time and fill direction breakdown.

**Adverse Selection** — four-panel stratification of markout by: queue position at entry, maker age at fill, volume depleted ahead before fill, and book imbalance at fill time. These are the structural variables that explain *where* the adverse selection is coming from.

**Pricing Signals & Toxicity** — two-panel fair value chart: microprice vs BBO mid vs OU fundamental on the price axis, imbalance EMA on a separate axis (different units — it's a signal, not a price). Toxicity EMA, adaptive spread widening, and the scatter of toxicity widen vs subsequent markout.

**LOB Depth** — stacked area chart of 5-level depth on each side, TOB imbalance over time, MM queue share at best bid/ask.

**Robustness** — 10-seed sweep of final equity, max drawdown, and equity vs drawdown scatter. A strategy that is profitable on one seed and not the others tells you something.

**Execution & Diagnostics** — latency distribution with log-scale x-axis and fitted log-normal overlay (linear scale collapses the tail into invisibility). Median, 95th, and 99th percentile lines. Hawkes intensity over time. Cancel latency vs flow intensity scatter for verifying the congestion mechanism. Accounting identity residual. Per-step P&L histogram and Q-Q plot.

---

## Running it

**Build and run the C++ engine:**
```bash
g++ -std=c++17 -O2 -o market-sim market-sim-v3.0.cpp
./market-sim
```

Writes `telemetry_v3_hawkes.csv`, `trades_v3_hawkes.csv`, `telemetry_v3_replay.csv`, `trades_v3_replay.csv`, and `multi_seed_summary_v3.csv`.

**Launch the viewer:**
```bash
pip install streamlit pandas numpy matplotlib seaborn scipy
streamlit run market_sim_viewer_v3_0.py
```

Upload the telemetry and trades CSVs from the sidebar. Multi-seed summary is optional.

**To connect real historical data:**  
Convert a Databento MBO or Binance L3 feed to the replay CSV schema:
```
t_s,event,side,price_ticks,qty,order_id
```
where `event` is `A` (add limit), `C` (cancel), or `M` (market order), and `t_s` is seconds since session open. Set `cfg.use_replay = true` and `cfg.replay_path = "your_feed.csv"` in `main()`.

---

## What this isn't

- Not a production backtester. There's no historical data ingestion by default, no real fill model, no slippage on the MM's own orders beyond queue position.
- Not a calibrated model. The Hawkes parameters and OU volatility are set to produce interesting dynamics, not to match a specific instrument. If you want to make claims about a real market, you need to calibrate against real data.
- Not annualised. The simulation runs for ~300 simulated seconds of a high-volatility instrument. The rolling IR in the viewer is a diagnostic ratio over that window. Do not annualise it.

---

## Why this exists

Built to understand market making by implementing the mechanics rather than reading about them: how FIFO queue priority works in practice, how inventory accumulates during one-sided flow, why a constant-latency cancel model is too optimistic, why fair value should come from the book and not a latent fundamental, and why you need to stratify adverse selection by queue position and depletion rather than just looking at aggregate P&L.

The comments in the C++ source are written to explain *why* each decision was made, not just what it does.
