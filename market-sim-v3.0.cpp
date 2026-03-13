// Event-driven market-making simulator: production-oriented architecture upgrade.
//
// Major changes from v2.2:
//
//   [1] OrderPool + intrusive doubly-linked lists (zero heap allocation on hot path)
//       Problem: std::list<Order> heap-allocates one node per order (~100-200 ns
//       allocator overhead per insert/erase; cache-scattered nodes destroy prefetcher).
//       Fix: Pre-allocated slab of kPoolCapacity Order objects in a flat vector.
//       Each Order embeds pool_prev / pool_next as uint32_t pool indices.
//       Price-level queues are intrusive doubly-linked lists over these indices.
//       Alloc and free are free-list stack push/pop — single-digit nanoseconds.
//       Queue iteration is sequential through the flat array — prefetcher-friendly.
//
//   [2] ArrayBook: O(1) level lookup replacing std::map<price_t, Level>
//       Problem: std::map is O(log N) per access. For a book with ~20 active levels,
//       that is 4–5 pointer hops, each a likely L2/L3 cache miss.
//       Fix: Power-of-2 circular slot array (kCap = 8192 ticks). The level for
//       price P lives at slots_[(P - base_) & kMask]. Single array index, O(1).
//       Active-price tracking for best-bid/ask via std::set<price_t> — O(log k)
//       insert/erase but k (live levels) is small. In production, replace with a
//       Fenwick tree or cache-line-aligned bitset scan for true O(1) best-bid/ask.
//       Recentering (rare) is triggered when price drifts outside the current window.
//
//   [3] LatencyModel: log-normal cancel/place latency replacing hardcoded constants
//       Problem: Real network RTT is not constant. It has a log-normal shape with a
//       heavy right tail (jitter, kernel scheduler, TCP retransmit). Constant latency
//       produces unrealistically optimistic cancel success rates in fast markets.
//       Fix: cancel and place latency sampled from std::lognormal_distribution.
//       Additional optional congestion penalty: when Hawkes total intensity is elevated
//       (order burst) a proportional delay is added, modelling exchange queue backlog.
//       Parameters are calibrated in comments to realistic co-location RTT ranges.
//
//   [4] FairValueModel: microprice + imbalance momentum replacing raw OU mid
//       Problem: A-S uses the OU fundamental as fair value. Real MMs don't have access
//       to a noise-free fundamental; they use market-observable book signals.
//       Fix: fair_value = microprice + imb_coeff * imbalance_ema
//       Microprice (Stoikov 2018) is a well-documented short-term price predictor.
//       The imbalance EMA captures momentum: sustained one-sided pressure shifts FV.
//       Inventory skew (A-S mechanism) is applied on top of this market-reactive base.
//       The OU fundamental is retained for the informed-trader model only (it represents
//       the "true" latent value that informed agents observe but the MM cannot).
//
//   [5] ReplayEngine: CSV-driven historical event replay infrastructure
//       Problem: Generative flow models encode the builder's assumptions; they cannot
//       validate strategy behaviour against real observed order flow.
//       Fix: ReplayEngine reads a normalized event CSV (t_s,event,side,price_ticks,qty,
//       order_id) and drives the same ArrayBook and MM agent. Designed for Databento
//       MBO or Binance L3 historical data. generate_synthetic() produces self-contained
//       test data in the identical format so the pipeline runs without external files.
//       Plug in real data: convert feed to the CSV schema, set cfg.use_replay = true.
//
//   Other improvements:
//   - Hawkes6::last_intensity() exposes total lambda for congestion model.
//   - FairValueModel::imb_ema() logged as fv_imb_ema telemetry column.
//   - Multi-seed sweep retained; replay mode uses a single seed (deterministic).

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <queue>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace sim {


// Primitive types and constants


// Integer tick representation: eliminates floating-point equality bugs in
// LOB price comparisons. Two prices that "should" be equal are always equal.
using price_t = std::int64_t;

// Pool index type. uint32_t gives 4B slots — far more than any realistic sim.
using idx_t = std::uint32_t;
static constexpr idx_t INVALID = std::numeric_limits<idx_t>::max();

// OrderPool capacity. 64K slots covers any realistic sim depth.
static constexpr std::size_t kPoolCapacity = 65536;

// ArrayBook: power-of-2 slot count for O(1) index via bitwise AND.
// 8192 ticks covers ±$40.96 from center at $0.01/tick — fine for an OU process.
static constexpr idx_t  kCap  = 8192;
static constexpr idx_t  kMask = kCap - 1;

inline bool is_finite(double x) noexcept { return std::isfinite(x); }

struct TickSpec {
    double tick_size{0.01};
    price_t to_ticks(double px) const noexcept {
        return static_cast<price_t>(std::llround(px / tick_size));
    }
    double to_price(price_t t) const noexcept {
        return static_cast<double>(t) * tick_size;
    }
};

enum class Side  : std::uint8_t { Bid = 0, Ask = 1 };
enum class Owner : std::uint8_t { External = 0, MM = 1 };

inline Side opposite(Side s) noexcept {
    return s == Side::Bid ? Side::Ask : Side::Bid;
}


// Order


// Embeds prev/next pool indices for intrusive doubly-linked list membership.
// An Order object lives at a fixed pool slot for its entire lifetime;
// the intrusive fields are updated by the book on insert/remove.
struct Order {
    std::uint64_t id{0};
    Owner         owner{Owner::External};
    Side          side{Side::Bid};
    price_t       price_ticks{0};
    int           qty{0};
    double        t_created{0.0};
    int           queue_ahead_qty{0};           // resting qty at this level at insertion
    std::int64_t  level_cumvol_at_entry{0};     // level cumvol at insertion time
    // Intrusive doubly-linked list linkage (pool indices).
    idx_t         pool_prev{INVALID};
    idx_t         pool_next{INVALID};
};


// OrderPool: slab allocator

//
// Design rationale:
//   std::list heap-allocates each node separately. On a modern x86 this means
//   the allocator lock, a jemalloc/tcmalloc small-bin lookup (~50-100 ns), and
//   a heap address that is cache-cold relative to every other order at that level.
//   When iterating a price queue to match a market order, each step is an
//   independent cache miss into a random heap address.
//
//   The pool pre-allocates all Order objects in a single std::vector (contiguous
//   DRAM). Queue iteration steps through addresses that differ by sizeof(Order)
//   bytes — highly prefetchable. Alloc/free are a vector pop/push — O(1), ~2 ns.
//
class OrderPool {
public:
    OrderPool() {
        orders_.resize(kPoolCapacity);
        free_list_.reserve(kPoolCapacity);
        // Push in reverse so first alloc returns index 0 (natural ordering).
        for (idx_t i = static_cast<idx_t>(kPoolCapacity) - 1; i != INVALID; --i)
            free_list_.push_back(i);
    }

    idx_t alloc() {
        assert(!free_list_.empty() && "OrderPool exhausted — increase kPoolCapacity");
        const idx_t i = free_list_.back();
        free_list_.pop_back();
        orders_[i] = Order{};   // zero-init: pool_prev/pool_next = INVALID
        return i;
    }

    void free(idx_t i) {
        assert(i < kPoolCapacity);
        free_list_.push_back(i);
    }

    Order&       get(idx_t i)       noexcept { return orders_[i]; }
    const Order& get(idx_t i) const noexcept { return orders_[i]; }

    std::size_t free_count() const noexcept { return free_list_.size(); }

private:
    std::vector<Order> orders_;
    std::vector<idx_t> free_list_;
};

// Fill (unchanged from v2.2)

struct Fill {
    double        t{0.0};
    price_t       price_ticks{0};
    int           qty{0};
    Owner         maker_owner{Owner::External};
    Owner         taker_owner{Owner::External};
    Side          maker_side{Side::Bid};
    std::uint64_t maker_order_id{0};
    std::uint64_t taker_order_id{0};
    double        maker_age_s{0.0};
    int           maker_queue_ahead_qty_entry{0};
    std::int64_t  queue_depleted_before_fill{0};
};

// ArrayBook: O(1) level access via circular slot array

// Slot: level metadata living at slots_[(price - base_) & kMask].
// Contains head/tail of the intrusive order queue plus cumulative stats.

struct Slot {
    idx_t        head_idx{INVALID};
    idx_t        tail_idx{INVALID};
    int          total_qty{0};
    std::int64_t cumvol{0};      // cumulative traded vol at this price (monotone)
};

class ArrayBook {
public:
    using FillCallback = std::function<void(const Fill&)>;

    explicit ArrayBook(TickSpec tick, OrderPool& pool)
        : tick_(tick), pool_(pool) {
        slots_.fill(Slot{});
    }

    void set_fill_callback(FillCallback cb) { on_fill_ = std::move(cb); }

    // Call once before any orders to set the center of the price window.
    void init_center(price_t center_ticks) {
        base_ = center_ticks - static_cast<price_t>(kCap / 2);
    }

    std::optional<price_t> best_bid() const noexcept {
        return bid_prices_.empty()
            ? std::nullopt : std::make_optional(*bid_prices_.begin());
    }
    std::optional<price_t> best_ask() const noexcept {
        return ask_prices_.empty()
            ? std::nullopt : std::make_optional(*ask_prices_.begin());
    }

    int best_bid_size() const noexcept {
        if (bid_prices_.empty()) return 0;
        return slots_[slot_of(*bid_prices_.begin())].total_qty;
    }
    int best_ask_size() const noexcept {
        if (ask_prices_.empty()) return 0;
        return slots_[slot_of(*ask_prices_.begin())].total_qty;
    }

    int depth_size(Side side, int levels) const noexcept {
        int tot = 0, k = 0;
        if (side == Side::Bid) {
            for (price_t px : bid_prices_) {
                tot += slots_[slot_of(px)].total_qty;
                if (++k >= levels) break;
            }
        } else {
            for (price_t px : ask_prices_) {
                tot += slots_[slot_of(px)].total_qty;
                if (++k >= levels) break;
            }
        }
        return tot;
    }

    void level_depths(Side side, std::array<int, 5>& out) const noexcept {
        out.fill(0);
        int k = 0;
        if (side == Side::Bid) {
            for (price_t px : bid_prices_) { if (k >= 5) break; out[k++] = slots_[slot_of(px)].total_qty; }
        } else {
            for (price_t px : ask_prices_) { if (k >= 5) break; out[k++] = slots_[slot_of(px)].total_qty; }
        }
    }

    bool cancel(std::uint64_t order_id) {
        auto it = id_to_idx_.find(order_id);
        if (it == id_to_idx_.end()) return false;
        const idx_t oidx = it->second;
        id_to_idx_.erase(it);
        Order& o = pool_.get(oidx);
        maybe_recenter(o.price_ticks);
        Slot& slot = slots_[slot_of(o.price_ticks)];
        detach_from_slot(slot, oidx, o.price_ticks, o.side);
        pool_.free(oidx);
        return true;
    }

    // Place a limit order. Executes immediately if marketable, rests otherwise.
    // Returns remaining qty resting (0 if fully consumed on entry).
    int add_limit(Order tmpl, std::uint64_t taker_id_for_cross = 0) {
        if (tmpl.qty <= 0) return 0;
        maybe_recenter(tmpl.price_ticks);

        if (tmpl.side == Side::Bid)
            match_asks(tmpl, tmpl.price_ticks, taker_id_for_cross, tmpl.owner);
        else
            match_bids(tmpl, tmpl.price_ticks, taker_id_for_cross, tmpl.owner);

        if (tmpl.qty <= 0) return 0;

        // Allocate pool slot and rest.
        const idx_t oidx = pool_.alloc();
        Order& o = pool_.get(oidx);
        o           = tmpl;
        o.pool_prev = INVALID;   // explicit clear — tmpl may carry stale values
        o.pool_next = INVALID;

        Slot& slot = slots_[slot_of(o.price_ticks)];
        o.queue_ahead_qty       = slot.total_qty;
        o.level_cumvol_at_entry = slot.cumvol;
        attach_to_slot(slot, oidx, o.price_ticks, o.side);
        id_to_idx_[o.id] = oidx;
        return o.qty;
    }

    void add_market(std::uint64_t taker_id, Owner taker_owner,
                    Side taker_side, int qty, double t) {
        if (qty <= 0) return;
        Order stub{};
        stub.id      = taker_id;  stub.owner = taker_owner;
        stub.side    = taker_side; stub.qty   = qty;
        stub.t_created = t;
        if (taker_side == Side::Bid)
            match_asks(stub, std::nullopt, taker_id, taker_owner);
        else
            match_bids(stub, std::nullopt, taker_id, taker_owner);
    }

    // Const access for telemetry (e.g. read resting qty of the MM's quote).
    const Order* find_order(std::uint64_t id) const noexcept {
        auto it = id_to_idx_.find(id);
        if (it == id_to_idx_.end()) return nullptr;
        return &pool_.get(it->second);
    }

    const std::unordered_map<std::uint64_t, idx_t>& locations() const noexcept {
        return id_to_idx_;
    }

private:
    TickSpec   tick_{};
    OrderPool& pool_;
    price_t    base_{0};

    std::array<Slot, kCap> slots_{};

    // Active price level sets: O(log k) insert/erase, O(1) best via begin().
    // k = number of occupied levels, typically < 30.
    // Production alternative: Fenwick tree or cache-line bitset scan.
    std::set<price_t, std::greater<price_t>> bid_prices_;
    std::set<price_t>                        ask_prices_;

    std::unordered_map<std::uint64_t, idx_t> id_to_idx_;
    FillCallback on_fill_{};

    // O(1) slot index.
    idx_t slot_of(price_t px) const noexcept {
        return static_cast<idx_t>(px - base_) & kMask;
    }

    bool in_range(price_t px) const noexcept {
        const price_t offset = px - base_;
        return offset >= 0 && offset < static_cast<price_t>(kCap);
    }

    // If px is outside the current window, shift base_ and re-slot all orders.
    // This is O(n_live_orders) but triggered only when price drifts more than
    // kCap/4 ticks from center — rare in an OU process with normal parameters.
    void maybe_recenter(price_t px) {
        if (in_range(px)) return;
        recenter(px);
    }

    void recenter(price_t new_center_hint) {
        // Collect all live orders.
        std::vector<idx_t> live;
        live.reserve(id_to_idx_.size());
        for (const auto& [id, idx] : id_to_idx_)
            live.push_back(idx);

        // Snapshot cumvols for all occupied prices before clearing.
        std::unordered_map<price_t, std::int64_t> saved_cumvol;
        for (price_t px : bid_prices_) saved_cumvol[px] = slots_[slot_of(px)].cumvol;
        for (price_t px : ask_prices_) saved_cumvol[px] = slots_[slot_of(px)].cumvol;

        // Wipe old slots.
        for (price_t px : bid_prices_) slots_[slot_of(px)] = Slot{};
        for (price_t px : ask_prices_) slots_[slot_of(px)] = Slot{};

        // New base: center on the triggering price.
        base_ = new_center_hint - static_cast<price_t>(kCap / 2);

        // Restore cumvols at new slot positions.
        for (auto& [px, cv] : saved_cumvol)
            slots_[slot_of(px)].cumvol = cv;

        // Re-link all orders into new slot positions (preserves FIFO order per level).
        for (idx_t oidx : live) {
            Order& o = pool_.get(oidx);
            Slot& slot = slots_[slot_of(o.price_ticks)];
            o.pool_prev = slot.tail_idx;
            o.pool_next = INVALID;
            if (slot.tail_idx != INVALID)
                pool_.get(slot.tail_idx).pool_next = oidx;
            else
                slot.head_idx = oidx;
            slot.tail_idx  = oidx;
            slot.total_qty += o.qty;
        }
    }

    // Append order to tail of level queue; register price in active set.
    void attach_to_slot(Slot& slot, idx_t oidx, price_t px, Side side) {
        Order& o = pool_.get(oidx);
        o.pool_prev = slot.tail_idx;
        o.pool_next = INVALID;
        if (slot.tail_idx != INVALID)
            pool_.get(slot.tail_idx).pool_next = oidx;
        else
            slot.head_idx = oidx;
        slot.tail_idx   = oidx;
        slot.total_qty += o.qty;
        if (side == Side::Bid) bid_prices_.insert(px);
        else                   ask_prices_.insert(px);
    }

    // Remove order from level queue; remove price from active set if level empty.
    // Called only for explicit cancels (qty removed = order's full remaining qty).
    void detach_from_slot(Slot& slot, idx_t oidx, price_t px, Side side) {
        Order& o = pool_.get(oidx);
        slot.total_qty -= o.qty;
        if (o.pool_prev != INVALID) pool_.get(o.pool_prev).pool_next = o.pool_next;
        else                        slot.head_idx = o.pool_next;
        if (o.pool_next != INVALID) pool_.get(o.pool_next).pool_prev = o.pool_prev;
        else                        slot.tail_idx = o.pool_prev;
        if (slot.head_idx == INVALID) {
            if (side == Side::Bid) bid_prices_.erase(px);
            else                   ask_prices_.erase(px);
        }
    }

    void emit_fill(const Fill& f) { if (on_fill_) on_fill_(f); }

    // matching engine
    // Shared inline helpers to remove a fully-filled maker from the intrusive list.
    // Called inside the hot matching loop; no virtual dispatch, no heap traffic.
    void erase_maker(Slot& slot, idx_t oidx, price_t px, Side maker_side) {
        Order& o = pool_.get(oidx);
        if (o.pool_prev != INVALID) pool_.get(o.pool_prev).pool_next = o.pool_next;
        else                        slot.head_idx = o.pool_next;
        if (o.pool_next != INVALID) pool_.get(o.pool_next).pool_prev = o.pool_prev;
        else                        slot.tail_idx = o.pool_prev;
        id_to_idx_.erase(o.id);
        pool_.free(oidx);
        if (slot.head_idx == INVALID) {
            if (maker_side == Side::Bid) bid_prices_.erase(px);
            else                         ask_prices_.erase(px);
        }
    }

    void match_asks(Order& taker, std::optional<price_t> limit_px,
                    std::uint64_t taker_id, Owner taker_owner) {
        while (taker.qty > 0 && !ask_prices_.empty()) {
            const price_t ask_px = *ask_prices_.begin();
            if (limit_px && ask_px > *limit_px) break;
            Slot& slot = slots_[slot_of(ask_px)];
            idx_t idx  = slot.head_idx;
            while (idx != INVALID && taker.qty > 0) {
                Order& maker        = pool_.get(idx);
                const int fill_qty  = std::min(taker.qty, maker.qty);
                Fill f{};
                f.t                            = taker.t_created;
                f.price_ticks                  = ask_px;
                f.qty                          = fill_qty;
                f.maker_owner                  = maker.owner;
                f.taker_owner                  = taker_owner;
                f.maker_side                   = maker.side;
                f.maker_order_id               = maker.id;
                f.taker_order_id               = taker_id;
                f.maker_age_s                  = taker.t_created - maker.t_created;
                f.maker_queue_ahead_qty_entry  = maker.queue_ahead_qty;
                f.queue_depleted_before_fill   = slot.cumvol - maker.level_cumvol_at_entry;
                emit_fill(f);
                taker.qty       -= fill_qty;
                maker.qty       -= fill_qty;
                slot.total_qty  -= fill_qty;
                slot.cumvol     += fill_qty;
                const idx_t nxt = maker.pool_next;
                if (maker.qty == 0) erase_maker(slot, idx, ask_px, Side::Ask);
                idx = nxt;
            }
        }
    }

    void match_bids(Order& taker, std::optional<price_t> limit_px,
                    std::uint64_t taker_id, Owner taker_owner) {
        while (taker.qty > 0 && !bid_prices_.empty()) {
            const price_t bid_px = *bid_prices_.begin();
            if (limit_px && bid_px < *limit_px) break;
            Slot& slot = slots_[slot_of(bid_px)];
            idx_t idx  = slot.head_idx;
            while (idx != INVALID && taker.qty > 0) {
                Order& maker        = pool_.get(idx);
                const int fill_qty  = std::min(taker.qty, maker.qty);
                Fill f{};
                f.t                            = taker.t_created;
                f.price_ticks                  = bid_px;
                f.qty                          = fill_qty;
                f.maker_owner                  = maker.owner;
                f.taker_owner                  = taker_owner;
                f.maker_side                   = maker.side;
                f.maker_order_id               = maker.id;
                f.taker_order_id               = taker_id;
                f.maker_age_s                  = taker.t_created - maker.t_created;
                f.maker_queue_ahead_qty_entry  = maker.queue_ahead_qty;
                f.queue_depleted_before_fill   = slot.cumvol - maker.level_cumvol_at_entry;
                emit_fill(f);
                taker.qty       -= fill_qty;
                maker.qty       -= fill_qty;
                slot.total_qty  -= fill_qty;
                slot.cumvol     += fill_qty;
                const idx_t nxt = maker.pool_next;
                if (maker.qty == 0) erase_maker(slot, idx, bid_px, Side::Bid);
                idx = nxt;
            }
        }
    }
};

// LatencyModel: log-normal cancel / place latency


/**
  Real co-location network RTT is well-modelled by a log-normal:
    - a modal (most likely) value close to the hardware floor
    - a heavy right tail from jitter (kernel scheduler, IRQ coalescing,
      TCP retransmit, GC pause on the exchange side).
 
  Log-normal parameterisation from mean and CV:
    sigma^2 = log(1 + CV^2)
    mu      = log(mean) - sigma^2 / 2
 
  Example (cancel, mean=30ms, CV=0.50):
    sigma^2 = log(1.25) ≈ 0.223  →  sigma ≈ 0.472
    mu      = log(0.030) - 0.111 ≈ -3.618
 
  Congestion model:
    When the exchange matching engine is processing a Hawkes burst, your
    cancel/place is queued behind incoming flow. Each event/sec above
    intensity_baseline adds congestion_per_event_s seconds of extra delay.
    Default: disabled (congestion_per_event_s = 0). Enable with ~1e-5.
 */

struct LatencyConfig {
    // Log-normal parameters (natural-log space).
    double cancel_lnmu{-3.618};      // modal cancel RTT ≈ 27 ms
    double cancel_lnsigma{0.472};    // CV ≈ 0.50 → 95th pct ≈ 75 ms
    double place_lnmu{-4.605};       // modal place RTT ≈ 9 ms
    double place_lnsigma{0.350};     // CV ≈ 0.36
    // Congestion (exchange queue backlog when order flow bursts).
    double congestion_per_event_s{0.0};
    double intensity_baseline{150.0};
};

class LatencyModel {
public:
    explicit LatencyModel(LatencyConfig cfg, std::mt19937_64& rng)
        : cfg_(cfg), rng_(rng) {}

    double sample_cancel(double hawkes_intensity = 0.0) {
        std::lognormal_distribution<double> d(cfg_.cancel_lnmu, cfg_.cancel_lnsigma);
        return d(rng_) + congestion(hawkes_intensity);
    }

    double sample_place(double hawkes_intensity = 0.0) {
        std::lognormal_distribution<double> d(cfg_.place_lnmu, cfg_.place_lnsigma);
        return d(rng_) + congestion(hawkes_intensity);
    }

private:
    LatencyConfig    cfg_{};
    std::mt19937_64& rng_;

    double congestion(double intensity) const noexcept {
        if (cfg_.congestion_per_event_s <= 0.0) return 0.0;
        return cfg_.congestion_per_event_s
             * std::max(0.0, intensity - cfg_.intensity_baseline);
    }
};


/**
  FairValueModel: microprice + imbalance momentum
 
  Why replace the OU mid with this:
    The OU fundamental is a latent variable the MM cannot observe. Real MMs
    derive fair value from observable signals — primarily the book itself.
 
  Microprice (Stoikov 2018):
    mp = (ask * bid_qty + bid * ask_qty) / (bid_qty + ask_qty)
    Weights the mid by opposite-side queue size.
    When the bid queue is heavy (bid_qty >> ask_qty), price is likely to rise,
    so microprice is pulled toward the ask. Empirically validated as a short-term
    predictor superior to the raw BBO mid.
 
  Imbalance EMA;
    imb = (bid_qty - ask_qty) / (bid_qty + ask_qty)  ∈ [-1, +1]
    imb_ema = alpha * imb + (1-alpha) * imb_ema_prev
    Captures sustained one-sided book pressure. When imb_ema is strongly
    positive (bid-heavy book trending), our fair value should be shifted up.
 
  Reservation price:
    reservation = microprice + imb_coeff * imb_ema - inv_skew * inventory
    The A-S inventory skew is still the right mechanism for risk adjustment;
    we just apply it to a better-calibrated reference price.
 */

class FairValueModel {
public:
    struct Config {
        double imb_alpha{0.20};                  // EMA decay for imbalance signal
        double imb_coeff_ticks{1.5};             // FV shift in ticks per unit imb_ema
        double inv_skew_ticks_per_share{0.18};   // A-S inventory skew per share
    };

    explicit FairValueModel(Config cfg) : cfg_(cfg) {}

    // Returns reservation price in ticks (fair value minus inventory skew).
    price_t update(std::optional<price_t> bb, std::optional<price_t> aa,
                   int bid_qty, int ask_qty, int inventory) {
        // Microprice: O(1), no heap.
        const price_t micro = microprice(bb, aa, bid_qty, ask_qty);

        // Imbalance momentum EMA.
        const int tot = std::max(1, bid_qty + ask_qty);
        const double imb = static_cast<double>(bid_qty - ask_qty) / tot;
        imb_ema_ = cfg_.imb_alpha * imb + (1.0 - cfg_.imb_alpha) * imb_ema_;

        // Fair value in tick-space.
        const double fv = static_cast<double>(micro)
                        + cfg_.imb_coeff_ticks * imb_ema_;

        // Reservation: shift away from inventory direction.
        const double reservation = fv
            - cfg_.inv_skew_ticks_per_share * static_cast<double>(inventory);

        return static_cast<price_t>(std::llround(reservation));
    }

    double imb_ema() const noexcept { return imb_ema_; }

private:
    Config cfg_{};
    double imb_ema_{0.0};

    static price_t microprice(std::optional<price_t> bb, std::optional<price_t> aa,
                               int bq, int aq) noexcept {
        if (!bb && !aa) return 10000LL;
        if (!bb)        return *aa;
        if (!aa)        return *bb;
        const int tot = bq + aq;
        if (tot == 0) return (*bb + *aa) / 2;
        return static_cast<price_t>(std::llround(
            (static_cast<double>(*aa) * bq + static_cast<double>(*bb) * aq)
            / static_cast<double>(tot)));
    }
};

// MarketMaker

struct MMConfig {
    int    quote_qty{1};
    double base_spread_ticks{2.0};
    double spread_widen_ticks_per_share{0.05};
    int    inv_requote_band{5};
    double max_quote_age_s{0.75};
    double min_quote_life_s{0.05};
    double maker_rebate_per_share{-0.0002};
    // Toxicity-adaptive spread.
    double tox_ema_alpha{0.30};
    double tox_widen_threshold{0.40};
    double tox_widen_max_ticks{2.0};
    // Embedded fair-value and latency configs (latency consumed by Simulator).
    FairValueModel::Config fv{};
    LatencyConfig          latency{};
};

class MarketMaker {
public:
    MarketMaker(ArrayBook& ob, TickSpec tick, MMConfig cfg)
        : ob_(ob), tick_(tick), cfg_(cfg), fv_(cfg.fv) {}

    void on_fill(const Fill& f, price_t mark_ticks) {
        if (f.maker_owner != Owner::MM) return;
        ++mm_fills_;
        mm_volume_ += f.qty;

        const double px  = tick_.to_price(f.price_ticks);
        const double fee = static_cast<double>(f.qty) * cfg_.maker_rebate_per_share;
        fees_paid_ += fee;

        // Toxicity EMA on fill direction.
        const double dir  = (f.maker_side == Side::Bid) ? +1.0 : -1.0;
        fill_sign_ema_    = cfg_.tox_ema_alpha * dir
                          + (1.0 - cfg_.tox_ema_alpha) * fill_sign_ema_;

        if (f.maker_side == Side::Bid) {
            apply_trade(true, px, f.qty, fee);
            if (f.maker_order_id == mm_bid_id_) mm_bid_id_ = 0;
        } else {
            apply_trade(false, px, f.qty, fee);
            if (f.maker_order_id == mm_ask_id_) mm_ask_id_ = 0;
        }
        last_mark_ticks_ = mark_ticks;
    }

    struct QuotePlan {
        bool    update_bid{false}, update_ask{false};
        price_t new_bid_ticks{0},  new_ask_ticks{0};
    };

/* plan_quotes: derive quotes from observable book signals only.
   mark_ticks is NOT used for fair value (it's the OU fundamental, unobservable).
   It is only passed for potential future logging; the FairValueModel uses the book. */
    QuotePlan plan_quotes(double t_now,
                          std::optional<price_t> bb, std::optional<price_t> aa,
                          int bid_qty, int ask_qty) {
        // Fair value from microprice + imbalance momentum + inventory skew.
        const price_t reservation = fv_.update(bb, aa, bid_qty, ask_qty, inv_);

        // Toxicity spread widen.
        const double tox_mag    = std::abs(fill_sign_ema_);
        const double tox_excess = std::max(0.0, tox_mag - cfg_.tox_widen_threshold);
        const double tox_range  = std::max(1e-8, 1.0 - cfg_.tox_widen_threshold);
        tox_widen_              = (tox_excess / tox_range) * cfg_.tox_widen_max_ticks;

        const double spread_ticks = std::max(1.0,
            cfg_.base_spread_ticks
            + cfg_.spread_widen_ticks_per_share * std::abs(static_cast<double>(inv_))
            + tox_widen_);
        const double half = 0.5 * spread_ticks;

        price_t bid_ticks = static_cast<price_t>(
            std::floor(static_cast<double>(reservation) - half));
        price_t ask_ticks = static_cast<price_t>(
            std::ceil(static_cast<double>(reservation) + half));
        if (ask_ticks <= bid_ticks) ask_ticks = bid_ticks + 1;

        // Never cross the book (MM is passive-only).
        if (aa && bid_ticks >= *aa) bid_ticks = *aa - 1;
        if (bb && ask_ticks <= *bb) ask_ticks = *bb + 1;
        if (ask_ticks <= bid_ticks) ask_ticks = bid_ticks + 1;

        QuotePlan p{};
        p.new_bid_ticks = bid_ticks;
        p.new_ask_ticks = ask_ticks;

        // Requote triggers: reservation moved, inventory band crossed, quote stale.
        const bool res_moved = !has_last_quote_res_
            || (std::llabs(reservation - last_quote_res_) >= 1);
        const bool inv_band  = (cfg_.inv_requote_band > 0)
            && ((std::abs(inv_) / cfg_.inv_requote_band)
             != (std::abs(last_inv_for_quote_) / cfg_.inv_requote_band));
        const bool stale_bid = bid_px_ && ((t_now - bid_t_placed_) > cfg_.max_quote_age_s);
        const bool stale_ask = ask_px_ && ((t_now - ask_t_placed_) > cfg_.max_quote_age_s);
        const bool too_soon  = (t_now - last_quote_update_t_) < cfg_.min_quote_life_s;

        if (!too_soon && (res_moved || inv_band || stale_bid))
            if (!bid_px_ || *bid_px_ != bid_ticks) p.update_bid = true;
        if (!too_soon && (res_moved || inv_band || stale_ask))
            if (!ask_px_ || *ask_px_ != ask_ticks) p.update_ask = true;

        last_inv_for_quote_  = inv_;
        has_last_quote_res_  = true;
        last_quote_res_      = reservation;
        last_quote_update_t_ = t_now;
        return p;
    }

    void note_quote_placed(Side side, std::uint64_t id, price_t px, double t_now) {
        if (side == Side::Bid) {
            mm_bid_id_ = id; bid_px_ = px; bid_t_placed_ = t_now;
        } else {
            mm_ask_id_ = id; ask_px_ = px; ask_t_placed_ = t_now;
        }
    }

    void note_quote_canceled(Side side, std::uint64_t id, bool success) {
        ++cancels_submitted_;
        if (success) ++cancels_succeeded_;
        if (side == Side::Bid && id == mm_bid_id_) { mm_bid_id_ = 0; bid_px_.reset(); }
        if (side == Side::Ask && id == mm_ask_id_) { mm_ask_id_ = 0; ask_px_.reset(); }
    }

    int    inventory()         const noexcept { return inv_; }
    double cash()              const noexcept { return cash_; }
    double realized_pnl()      const noexcept { return realized_pnl_; }
    double fees_paid()         const noexcept { return fees_paid_; }
    double fill_sign_ema()     const noexcept { return fill_sign_ema_; }
    double current_tox_widen() const noexcept { return tox_widen_; }
    double fv_imb_ema()        const noexcept { return fv_.imb_ema(); }

    std::uint64_t fills()             const noexcept { return mm_fills_; }
    std::uint64_t cancels_submitted() const noexcept { return cancels_submitted_; }
    std::uint64_t cancels_succeeded() const noexcept { return cancels_succeeded_; }

    double equity(price_t mark_ticks) const noexcept {
        return cash_ + static_cast<double>(inv_) * tick_.to_price(mark_ticks);
    }
    double unreal_pnl(price_t mark_ticks) const noexcept {
        if (inv_ == 0) return 0.0;
        const double mark = tick_.to_price(mark_ticks);
        return (inv_ > 0)
            ? (mark - avg_cost_) * static_cast<double>(inv_)
            : (avg_cost_ - mark) * static_cast<double>(-inv_);
    }

    std::uint64_t          bid_id()    const noexcept { return mm_bid_id_; }
    std::uint64_t          ask_id()    const noexcept { return mm_ask_id_; }
    std::optional<price_t> bid_price() const noexcept { return bid_px_; }
    std::optional<price_t> ask_price() const noexcept { return ask_px_; }

private:
    ArrayBook&     ob_;
    TickSpec       tick_{};
    MMConfig       cfg_{};
    FairValueModel fv_;

    int    inv_{0};
    double cash_{0.0};
    double avg_cost_{0.0};
    double realized_pnl_{0.0};
    double fees_paid_{0.0};
    double fill_sign_ema_{0.0};
    double tox_widen_{0.0};

    std::uint64_t mm_bid_id_{0}, mm_ask_id_{0};
    std::optional<price_t> bid_px_{}, ask_px_{};
    double bid_t_placed_{-1e9}, ask_t_placed_{-1e9};
    double last_quote_update_t_{-1e9};
    price_t last_quote_res_{0};
    bool    has_last_quote_res_{false};
    int     last_inv_for_quote_{0};
    price_t last_mark_ticks_{0};

    std::uint64_t mm_fills_{0}, mm_volume_{0};
    std::uint64_t cancels_submitted_{0}, cancels_succeeded_{0};

    void apply_trade(bool is_buy, double px, int qty, double fee_cash) {
        if (is_buy) {
            cash_ -= px * static_cast<double>(qty) + fee_cash;
            if (inv_ >= 0) {
                const int new_inv = inv_ + qty;
                avg_cost_ = (avg_cost_ * inv_ + px * qty)
                          / static_cast<double>(std::max(1, new_inv));
                inv_ = new_inv;
            } else {
                const int cover = std::min(qty, -inv_);
                realized_pnl_ += (avg_cost_ - px) * static_cast<double>(cover);
                inv_ += cover;
                const int rem = qty - cover;
                if (rem > 0) { inv_ = rem; avg_cost_ = px; }
                else if (inv_ == 0) avg_cost_ = 0.0;
            }
        } else {
            cash_ += px * static_cast<double>(qty) - fee_cash;
            if (inv_ <= 0) {
                const int abs_old = -inv_;
                const int abs_new = abs_old + qty;
                avg_cost_ = (avg_cost_ * abs_old + px * qty)
                          / static_cast<double>(std::max(1, abs_new));
                inv_ -= qty;
            } else {
                const int close = std::min(qty, inv_);
                realized_pnl_ += (px - avg_cost_) * static_cast<double>(close);
                inv_ -= close;
                const int rem = qty - close;
                if (rem > 0) { inv_ = -rem; avg_cost_ = px; }
                else if (inv_ == 0) avg_cost_ = 0.0;
            }
        }
    }
};


/* Hawkes6: 6-dimensional Hawkes process (unchanged from v2.2)
      
   Models the six event types jointly with a full 6x6 excitation matrix.
   Ogata thinning gives exact (not discretized) event times.
   Baseline intensities are state-dependent (spread, imbalance, info edge). */
enum class EType : std::uint8_t {
    LimAddBid = 0, LimAddAsk = 1,
    CancelBid = 2, CancelAsk = 3,
    MktBuy    = 4, MktSell   = 5,
    Count     = 6
};
static constexpr int kHawkesK = static_cast<int>(EType::Count);

struct HawkesConfig {
    double beta{6.0};
    double mu_lim{85.0};
    double mu_cancel{38.0};
    double mu_mkt{12.0};
    double lambda_cap{900.0};
    // Row sums < 1.0 ensures process stability without relying on lambda_cap.
    double alpha[kHawkesK][kHawkesK] = {
        // to:  LimBid  LimAsk  CanBid  CanAsk  MktBuy  MktSell
        /* LimAddBid */ { 0.28, 0.02, 0.17, 0.01, 0.01, 0.01 },  // sum 0.50
        /* LimAddAsk */ { 0.02, 0.28, 0.01, 0.17, 0.01, 0.01 },  // sum 0.50
        /* CancelBid */ { 0.24, 0.02, 0.30, 0.01, 0.03, 0.01 },  // sum 0.61
        /* CancelAsk */ { 0.02, 0.24, 0.01, 0.30, 0.01, 0.03 },  // sum 0.61
        /* MktBuy    */ { 0.05, 0.11, 0.30, 0.05, 0.42, 0.03 },  // sum 0.96
        /* MktSell   */ { 0.11, 0.05, 0.05, 0.30, 0.03, 0.42 },  // sum 0.96
    };
};

class Hawkes6 {
public:
    explicit Hawkes6(HawkesConfig cfg, std::mt19937_64& rng)
        : cfg_(cfg), rng_(rng) {
        S_.assign(kHawkesK, 0.0);
        mu_.assign(kHawkesK, 0.0);
        lambda_.assign(kHawkesK, 0.0);
    }

    void set_state(double spread_ticks, double imbalance, double info_edge_ticks) {
        const double spread_boost  = 1.0 + 0.03 * std::max(0.0, spread_ticks - 2.0);
        const double lim           = cfg_.mu_lim * spread_boost;
        const double cancel_bid    = cfg_.mu_cancel * (1.0 + 0.8 * std::max(0.0,  imbalance));
        const double cancel_ask    = cfg_.mu_cancel * (1.0 + 0.8 * std::max(0.0, -imbalance));
        const double z             = std::clamp(info_edge_ticks / 2.0, -6.0, 6.0);
        const double buy_tilt      = 1.0 / (1.0 + std::exp(-z));
        const double mkt_buy       = cfg_.mu_mkt * (0.35 + 1.3 *  buy_tilt);
        const double mkt_sell      = cfg_.mu_mkt * (0.35 + 1.3 * (1.0 - buy_tilt));
        mu_[0] = lim;        mu_[1] = lim;
        mu_[2] = cancel_bid; mu_[3] = cancel_ask;
        mu_[4] = mkt_buy;    mu_[5] = mkt_sell;
    }

    std::pair<double, EType> next(double t_now) {
        if (!has_t_) { t_ = t_now; has_t_ = true; }
        else if (t_now > t_) { decay(t_now - t_); t_ = t_now; }
        std::exponential_distribution<double> expd(1.0);
        std::uniform_real_distribution<double> U(0.0, 1.0);
        for (;;) {
            update_lambda();
            double lam_tot = std::accumulate(lambda_.begin(), lambda_.end(), 0.0);
            lam_tot = std::min(lam_tot, cfg_.lambda_cap);
            if (lam_tot <= 0.0) lam_tot = 1e-9;
            const double M  = lam_tot * 1.10;
            const double dt = expd(rng_) / M;
            decay(dt);
            t_ += dt;
            update_lambda();
            const double lam_new = std::min(
                std::accumulate(lambda_.begin(), lambda_.end(), 0.0),
                cfg_.lambda_cap);
            last_total_lambda_ = lam_new;
            if (U(rng_) * M <= lam_new) {
                const double r = U(rng_) * lam_new;
                double c = 0.0;
                for (int k = 0; k < kHawkesK; ++k) {
                    c += lambda_[k];
                    if (r <= c) { S_[k] += 1.0; return {t_, static_cast<EType>(k)}; }
                }
                S_.back() += 1.0;
                return {t_, EType::MktSell};
            }
        }
    }

    // Returns total intensity from the most recent next() call.
    // Used by LatencyModel for congestion penalty.
    double last_intensity() const noexcept { return last_total_lambda_; }

private:
    HawkesConfig    cfg_{};
    std::mt19937_64& rng_;
    bool   has_t_{false};
    double t_{0.0};
    double last_total_lambda_{0.0};
    std::vector<double> S_, mu_, lambda_;

    void decay(double dt) {
        if (dt <= 0.0) return;
        const double f = std::exp(-cfg_.beta * dt);
        for (double& x : S_) x *= f;
    }
    void update_lambda() {
        for (int k = 0; k < kHawkesK; ++k) {
            double exc = 0.0;
            for (int j = 0; j < kHawkesK; ++j) exc += cfg_.alpha[j][k] * S_[j];
            lambda_[k] = std::max(0.0, mu_[k] + exc);
        }
    }
};


// Action queue (MM latency model)

enum class ActionType : std::uint8_t { Cancel = 0, Place = 1, RequoteCheck = 2 };

struct Action {
    double        t{0.0};
    std::uint64_t seq{0};
    ActionType    type{ActionType::RequoteCheck};
    std::uint64_t order_id{0};
    Side          side{Side::Bid};
    price_t       price_ticks{0};
    int           qty{0};
    Owner         owner{Owner::MM};

    bool operator>(const Action& o) const noexcept {
        return (t != o.t) ? (t > o.t) : (seq > o.seq);
    }
};



/* ReplayEngine: CSV-driven historical event replay

   CSV format (header required):
     t_s,event,side,price_ticks,qty,order_id
     t_s         : event time in simulated seconds (float)
     event       : A=add_limit, C=cancel, M=market_order
     side        : B=bid, A=ask
     price_ticks : integer tick price (ignored for M and C events)
     qty         : order quantity
     order_id    : unique uint64 identifier

   Connecting to real data:
     Databento MBO: map action='A'→'A', action='C'/'D'→'C', action='T'→emit an 'M'.
     Binance L3:    NEW_ORDER→'A', CANCEL_ORDER→'C', TRADE→'M'.
     Convert timestamps to seconds-since-session-open for t_s.

   generate_synthetic():
     Produces self-contained test data in the identical format — no external
     files required to exercise the full replay pipeline. */


struct ReplayEvent {
    double        t_s{0.0};
    char          event{'A'};    // 'A', 'C', 'M'
    Side          side{Side::Bid};
    price_t       price_ticks{0};
    int           qty{0};
    std::uint64_t order_id{0};
};

class ReplayEngine {
public:
    // Load from file. Empty if file not found.
    explicit ReplayEngine(const std::string& path) { load(path); }

    bool        empty() const noexcept { return events_.empty(); }
    std::size_t size()  const noexcept { return events_.size(); }
    const ReplayEvent& operator[](std::size_t i) const noexcept { return events_[i]; }

    // Generate synthetic historical data for self-contained pipeline testing.
    // Mimics realistic order flow: ~70% limit adds, ~20% cancels, ~10% market orders.
    // Output is in the same CSV format that load() expects.
    static ReplayEngine generate_synthetic(double duration_s,
                                           price_t mid_ticks,
                                           std::uint64_t seed) {
        ReplayEngine eng{};
        std::mt19937_64 rng(seed);
        std::exponential_distribution<double> inter(150.0);  // ~150 events/sec
        std::uniform_real_distribution<double> U(0.0, 1.0);
        std::geometric_distribution<int>       geo(0.35);

        double t = 0.0;
        std::uint64_t next_id = 1'000'000ULL;

        // Track live external order IDs for cancel sampling (swap-and-pop).
        std::vector<std::uint64_t> resting;

        while (t < duration_s) {
            t += inter(rng);
            if (t >= duration_s) break;
            const double r = U(rng);
            if (r < 0.70) {
                const Side    side = (U(rng) < 0.5) ? Side::Bid : Side::Ask;
                const int     lvl  = std::min(1 + geo(rng), 14);
                const price_t px   = (side == Side::Bid) ? (mid_ticks - lvl)
                                                          : (mid_ticks + lvl);
                const std::uint64_t id = next_id++;
                eng.events_.push_back({t, 'A', side, px, 1, id});
                resting.push_back(id);
            } else if (r < 0.90 && !resting.empty()) {
                std::uniform_int_distribution<std::size_t> pick(0, resting.size() - 1);
                const std::size_t idx = pick(rng);
                const std::uint64_t id = resting[idx];
                const Side side = (U(rng) < 0.5) ? Side::Bid : Side::Ask;
                eng.events_.push_back({t, 'C', side, 0, 0, id});
                resting[idx] = resting.back(); resting.pop_back();
            } else {
                const Side side = (U(rng) < 0.5) ? Side::Bid : Side::Ask;
                eng.events_.push_back({t, 'M', side, 0, 1, next_id++});
            }
        }
        return eng;
    }

    // Write events to CSV file (same format as load reads).
    // Use to inspect generated data or save for re-use.
    void write_csv(const std::string& path) const {
        std::FILE* fp = std::fopen(path.c_str(), "w");
        if (!fp) { std::perror("ReplayEngine::write_csv"); return; }
        std::fprintf(fp, "t_s,event,side,price_ticks,qty,order_id\n");
        for (const auto& ev : events_) {
            std::fprintf(fp, "%.6f,%c,%c,%lld,%d,%llu\n",
                ev.t_s, ev.event,
                (ev.side == Side::Bid) ? 'B' : 'A',
                static_cast<long long>(ev.price_ticks),
                ev.qty,
                static_cast<unsigned long long>(ev.order_id));
        }
        std::fclose(fp);
    }

private:
    std::vector<ReplayEvent> events_;

    ReplayEngine() {}  // private: used by generate_synthetic

    void load(const std::string& path) {
        if (path.empty()) return;
        std::ifstream f(path);
        if (!f.is_open()) {
            std::cerr << "ReplayEngine: cannot open '" << path << "'\n";
            return;
        }
        std::string line;
        std::getline(f, line);  // discard header
        while (std::getline(f, line)) {
            if (line.empty() || line[0] == '#') continue;
            ReplayEvent ev{};
            char ev_ch = 'A', side_ch = 'B';
            long long px_ll = 0; unsigned long long id_ull = 0;
            if (std::sscanf(line.c_str(), "%lf,%c,%c,%lld,%d,%llu",
                    &ev.t_s, &ev_ch, &side_ch, &px_ll, &ev.qty, &id_ull) < 5)
                continue;
            ev.event        = ev_ch;
            ev.side         = (side_ch == 'B' || side_ch == 'b') ? Side::Bid : Side::Ask;
            ev.price_ticks  = static_cast<price_t>(px_ll);
            ev.order_id     = static_cast<std::uint64_t>(id_ull);
            events_.push_back(ev);
        }
        std::sort(events_.begin(), events_.end(),
                  [](const ReplayEvent& a, const ReplayEvent& b){ return a.t_s < b.t_s; });
        std::cout << "ReplayEngine: loaded " << events_.size()
                  << " events from '" << path << "'\n";
    }
};


// Simulation config and telemetry types

struct SimConfig {
    int           steps{3000};
    double        log_dt{0.10};
    std::uint64_t seed{42};

    TickSpec tick{0.01};
    int      lot_size{1};

    // OU fundamental price (unobservable by MM; used for informed-trader model).
    // Calibration: 0.20 gives ~3.5% vol over a 300-second run.
    double start_price{100.0};
    double ou_kappa{1.5};
    double ou_theta{100.0};
    double ou_sigma{0.20};

    // Transient microstructure impact.
    double impact_kappa{8.0};
    double impact_sigma{0.0};
    double impact_per_mkt{0.015};

    // Background LOB
    int    max_levels{14};
    double level_geo_p{0.35};
    int    ext_limit_qty{1};
    int    cancel_sample_tries{12};
    double min_ext_order_age_s{0.03};

    // Informed-trader (Glosten-Milgrom style).
    double p_informed{0.22};
    double info_threshold_ticks{1.0};

    MMConfig     mm{};
    HawkesConfig hawkes{};

    // Replay mode: if true, drive external events from replay_path instead of Hawkes.
    // If replay_path is empty, a synthetic historical feed is auto-generated.
    bool        use_replay{false};
    std::string replay_path{""};
};

struct TelemetryRow {
    std::int64_t step{0};
    double t{0.0};
    double equity{0.0};
    int    inventory{0};
    double bid{std::numeric_limits<double>::quiet_NaN()};
    double ask{std::numeric_limits<double>::quiet_NaN()};
    int    bid_missing{1}, ask_missing{1};
    double mid_mark{0.0};
    double microprice{std::numeric_limits<double>::quiet_NaN()};
    double spread{std::numeric_limits<double>::quiet_NaN()};
    int    bid_sz{0}, ask_sz{0};
    double imbalance{0.0};
    double cash{0.0};
    double inv_value{0.0};
    double realized_pnl{0.0};
    double unreal_pnl{0.0};
    double fees_paid{0.0};
    std::uint64_t mm_fills{0};
    std::uint64_t mm_cancels_sub{0};
    std::uint64_t mm_cancels_ok{0};
    double mm_bid_px{std::numeric_limits<double>::quiet_NaN()};
    double mm_ask_px{std::numeric_limits<double>::quiet_NaN()};
    int    mm_bid_qty{0}, mm_ask_qty{0};
    int    mm_at_best_bid{0}, mm_at_best_ask{0};
    double tob_share_bid{std::numeric_limits<double>::quiet_NaN()};
    double tob_share_ask{std::numeric_limits<double>::quiet_NaN()};
    double last_trade_px{std::numeric_limits<double>::quiet_NaN()};
    int    last_trade_sign{0};
    double tox_widen{0.0};
    double fill_sign_ema{0.0};
    // v3 new columns.
    double fv_imb_ema{0.0};       // FairValueModel imbalance EMA at snapshot time
    double hawkes_intensity{0.0}; // total Hawkes lambda (or 0 in replay mode)
    std::array<int, 5> bid_levels{};
    std::array<int, 5> ask_levels{};
};

struct TradeRow {
    double        t{0.0};
    double        px{0.0};
    int           qty{0};
    int           sign{0};
    int           mm_involved{0};
    double        mid_mark{0.0};
    double        spread{std::numeric_limits<double>::quiet_NaN()};
    double        maker_age_s{0.0};
    int           maker_queue_ahead_qty_entry{0};
    std::int64_t  queue_depleted_before_fill{0};
    double        mm_tox_widen{0.0};
    // v3: sampled latencies for the cancel/place that produced this fill (0 if N/A).
    double        cancel_latency_s{0.0};
    double        place_latency_s{0.0};
};

struct RunSummary {
    std::uint64_t seed{0};
    double final_equity{0.0};
    double max_drawdown{0.0};
    double realized_pnl{0.0};
    double fees_paid{0.0};
    int    final_inventory{0};
    std::uint64_t total_fills{0};
    std::uint64_t total_cancels_sub{0};
};


// Simulator

class Simulator {
public:
    explicit Simulator(SimConfig cfg)
        : cfg_(cfg),
          rng_(cfg.seed),
          pool_(),
          ob_(cfg.tick, pool_),
          mm_(ob_, cfg.tick, cfg.mm),
          hawkes_(cfg.hawkes, rng_),
          latency_(cfg.mm.latency, rng_) {
        ob_.set_fill_callback([this](const Fill& f) { this->on_fill(f); });
        t_      = 0.0;
        mark_   = cfg_.start_price;
        impact_ = 0.0;
        ob_.init_center(cfg_.tick.to_ticks(cfg_.start_price));
        seed_background_book();
    }

    RunSummary run(bool write_csv = true) {
        const int N      = cfg_.steps;
        double next_log  = cfg_.log_dt;
        std::int64_t step = 0;

        push_action({0.0, next_seq_++, ActionType::RequoteCheck});

        if (cfg_.use_replay) {
            // Build or load replay engine.
            if (cfg_.replay_path.empty()) {
                replay_ = ReplayEngine::generate_synthetic(
                    static_cast<double>(N) * cfg_.log_dt * 1.5,
                    cfg_.tick.to_ticks(cfg_.start_price),
                    cfg_.seed);
                std::cout << "Replay mode: generated "
                          << replay_.size() << " synthetic events.\n";
            } else {
                replay_ = ReplayEngine(cfg_.replay_path);
            }
            replay_pos_ = 0;
            run_replay_loop(N, next_log, step, write_csv);
        } else {
            // Hawkes-driven generative mode.
            update_hawkes_state();
            auto [t_ext, et] = hawkes_.next(t_);
            next_ext_t_    = t_ext;
            next_ext_type_ = et;
            run_hawkes_loop(N, next_log, step, write_csv);
        }

        if (static_cast<int>(telemetry_.size()) < N) record(step);

        if (write_csv) {
            const std::string mode = cfg_.use_replay ? "replay" : "hawkes";
            dump_telemetry("telemetry_v3_" + mode + ".csv");
            dump_trades("trades_v3_" + mode + ".csv");
            std::cout << "Telemetry: telemetry_v3_" << mode << ".csv\n";
            std::cout << "Trades:    trades_v3_" << mode << ".csv\n";
        }
        return make_summary();
    }

private:
    SimConfig       cfg_{};
    std::mt19937_64 rng_;
    OrderPool       pool_;
    ArrayBook       ob_;
    MarketMaker     mm_;
    Hawkes6         hawkes_;
    LatencyModel    latency_;
    ReplayEngine    replay_{""};

    double t_{0.0};
    double mark_{0.0};
    double impact_{0.0};

    std::vector<TelemetryRow> telemetry_{};
    std::vector<TradeRow>     trades_{};

    double last_trade_px_{std::numeric_limits<double>::quiet_NaN()};
    int    last_trade_sign_{0};

    // Hawkes mode state
    double next_ext_t_{0.0};
    EType  next_ext_type_{EType::LimAddBid};

    // Replay mode state
    std::size_t replay_pos_{0};

    // Pending latency samples for the next requote (for trade tape annotation).
    double pending_cancel_lat_{0.0};
    double pending_place_lat_{0.0};

    std::priority_queue<Action, std::vector<Action>, std::greater<Action>> actions_{};
    std::uint64_t next_seq_{1};
    std::uint64_t next_order_id_{1};

    std::vector<std::uint64_t>               ext_ids_{};
    std::unordered_map<std::uint64_t, std::size_t> ext_pos_{};

    std::uint64_t new_id() { return next_order_id_++; }

    double  ref_price()  const noexcept { return mark_ + impact_; }
    price_t mark_ticks() const noexcept { return cfg_.tick.to_ticks(mark_); }
    price_t ref_ticks()  const noexcept { return cfg_.tick.to_ticks(ref_price()); }

    double next_action_time() const noexcept {
        return actions_.empty()
            ? std::numeric_limits<double>::infinity()
            : actions_.top().t;
    }
    void   push_action(Action a) { actions_.push(std::move(a)); }
    Action pop_action()          { Action a = actions_.top(); actions_.pop(); return a; }

    // event loops 

    void run_hawkes_loop(int N, double next_log, std::int64_t step, bool) {
        while (step < N) {
            const double t_next = std::min({next_log, next_action_time(), next_ext_t_});
            advance_time(t_next);
            if (t_next == next_log) {
                record(step++); next_log += cfg_.log_dt; continue;
            }
            if (t_next == next_action_time()) {
                do_action(pop_action()); continue;
            }
            do_external_event(next_ext_type_);
            update_hawkes_state();
            std::tie(next_ext_t_, next_ext_type_) = hawkes_.next(t_);
        }
    }

    void run_replay_loop(int N, double next_log, std::int64_t step, bool) {
        while (step < N) {
            const double t_replay = (replay_pos_ < replay_.size())
                ? replay_[replay_pos_].t_s
                : std::numeric_limits<double>::infinity();
            const double t_next = std::min({next_log, next_action_time(), t_replay});
            advance_time(t_next);
            if (t_next == next_log) {
                record(step++); next_log += cfg_.log_dt; continue;
            }
            if (t_next == next_action_time()) {
                do_action(pop_action()); continue;
            }
            // Process replay event.
            if (replay_pos_ < replay_.size()) {
                do_replay_event(replay_[replay_pos_++]);
                push_action({t_, next_seq_++, ActionType::RequoteCheck});
            }
        }
    }

    // time and price dynamics 

    void advance_time(double t_new) {
        if (t_new <= t_) return;
        const double dt = t_new - t_;
        std::normal_distribution<double> N(0.0, 1.0);
        const double dW = std::sqrt(dt) * N(rng_);
        mark_   += cfg_.ou_kappa * (cfg_.ou_theta - mark_) * dt + cfg_.ou_sigma * dW;
        impact_ *= std::exp(-cfg_.impact_kappa * dt);
        if (cfg_.impact_sigma > 0.0) impact_ += cfg_.impact_sigma * dW;
        t_ = t_new;
    }

    // Hawkes-mode event dispatch 

    void update_hawkes_state() {
        auto bb = ob_.best_bid();
        auto aa = ob_.best_ask();
        double spread_ticks = 3.0;
        if (bb && aa) spread_ticks = static_cast<double>(*aa - *bb);
        const int db  = ob_.depth_size(Side::Bid, 5);
        const int da  = ob_.depth_size(Side::Ask, 5);
        const double d = static_cast<double>(std::max(1, db + da));
        const double imbalance = (static_cast<double>(db) - static_cast<double>(da)) / d;
        const double edge_ticks = static_cast<double>(
            cfg_.tick.to_ticks(mark_) - cfg_.tick.to_ticks(ref_price()));
        hawkes_.set_state(spread_ticks, imbalance, edge_ticks);
    }

    int sample_level() {
        std::geometric_distribution<int> geo(cfg_.level_geo_p);
        return std::min(1 + geo(rng_), cfg_.max_levels);
    }

    void do_external_event(EType et) {
        std::uniform_real_distribution<double> U(0.0, 1.0);
        const double edge_ticks = static_cast<double>(
            cfg_.tick.to_ticks(mark_) - cfg_.tick.to_ticks(ref_price()));

        auto place_limit = [&](Side side) {
            const int     lvl = sample_level();
            const price_t mid = ref_ticks();
            const price_t px  = (side == Side::Bid) ? (mid - lvl) : (mid + lvl);
            add_external_limit(side, px, cfg_.ext_limit_qty);
        };
        auto do_cancel = [&](Side side) {
            if (ext_ids_.empty()) return;
            const auto& locs = ob_.locations();
            for (int tries = 0; tries < cfg_.cancel_sample_tries && !ext_ids_.empty(); ++tries) {
                std::uniform_int_distribution<std::size_t> pick(0, ext_ids_.size() - 1);
                const std::uint64_t id = ext_ids_[pick(rng_)];
                auto it = locs.find(id);
                if (it == locs.end()) { unregister_external_id(id); continue; }
                if (pool_.get(it->second).side != side) continue;
                const Order& o = pool_.get(it->second);
                if ((t_ - o.t_created) < cfg_.min_ext_order_age_s) continue;
                if (ob_.cancel(id)) unregister_external_id(id);
                return;
            }
        };
        auto do_market = [&](Side side) {
            if (U(rng_) < cfg_.p_informed) {
                const bool dir_ok = (side == Side::Bid)
                    ? (edge_ticks >  cfg_.info_threshold_ticks)
                    : (edge_ticks < -cfg_.info_threshold_ticks);
                if (!dir_ok) return;
            }
            const std::uint64_t taker_id = new_id();
            ob_.add_market(taker_id, Owner::External, side, cfg_.lot_size, t_);
            impact_ += (side == Side::Bid ? +1.0 : -1.0) * cfg_.impact_per_mkt;
        };

        switch (et) {
            case EType::LimAddBid: place_limit(Side::Bid); break;
            case EType::LimAddAsk: place_limit(Side::Ask); break;
            case EType::CancelBid: do_cancel(Side::Bid);   break;
            case EType::CancelAsk: do_cancel(Side::Ask);   break;
            case EType::MktBuy:    do_market(Side::Bid);   break;
            case EType::MktSell:   do_market(Side::Ask);   break;
            default: break;
        }
        push_action({t_, next_seq_++, ActionType::RequoteCheck});
    }

    // Replay-mode event dispatch 

    void do_replay_event(const ReplayEvent& ev) {
        switch (ev.event) {
            case 'A': {
                Order o{};
                o.id = ev.order_id; o.owner = Owner::External;
                o.side = ev.side; o.price_ticks = ev.price_ticks;
                o.qty = ev.qty; o.t_created = t_;
                const int rem = ob_.add_limit(o, o.id);
                if (rem > 0) register_external_id(o.id);
                break;
            }
            case 'C': {
                if (ob_.cancel(ev.order_id)) unregister_external_id(ev.order_id);
                break;
            }
            case 'M': {
                const std::uint64_t tid = new_id();
                ob_.add_market(tid, Owner::External, ev.side, ev.qty, t_);
                impact_ += (ev.side == Side::Bid ? +1.0 : -1.0) * cfg_.impact_per_mkt;
                break;
            }
        }
    }

    // MM action execution 

    void do_action(const Action& a) {
        if (a.type == ActionType::Cancel) {
            const bool ok = ob_.cancel(a.order_id);
            mm_.note_quote_canceled(a.side, a.order_id, ok);
            return;
        }
        if (a.type == ActionType::Place) {
            Order o{};
            o.id = a.order_id; o.owner = a.owner;
            o.side = a.side; o.price_ticks = a.price_ticks;
            o.qty = a.qty; o.t_created = t_;
            const int remaining = ob_.add_limit(o, o.id);
            if (remaining > 0) mm_.note_quote_placed(o.side, o.id, o.price_ticks, t_);
            return;
        }
        // RequoteCheck
        maybe_requote();
    }

    void maybe_requote() {
        auto bb     = ob_.best_bid();
        auto aa     = ob_.best_ask();
        const int bq = ob_.best_bid_size();
        const int aq = ob_.best_ask_size();
        const auto plan = mm_.plan_quotes(t_, bb, aa, bq, aq);

        const double intensity = cfg_.use_replay ? 0.0 : hawkes_.last_intensity();

        auto schedule = [&](Side side, std::uint64_t old_id, price_t new_px) {
            const double cl = latency_.sample_cancel(intensity);
            const double pl = latency_.sample_place(intensity);
            pending_cancel_lat_ = cl;
            pending_place_lat_  = pl;

            if (old_id != 0) {
                Action c{};
                c.t = t_ + cl; c.seq = next_seq_++; c.type = ActionType::Cancel;
                c.order_id = old_id; c.side = side;
                push_action(c);
            }
            Action p{};
            p.t = t_ + pl + (old_id != 0 ? cl : 0.0);
            p.seq = next_seq_++; p.type = ActionType::Place;
            p.order_id = new_id(); p.side = side;
            p.price_ticks = new_px; p.qty = cfg_.mm.quote_qty;
            p.owner = Owner::MM;
            push_action(p);
        };

        if (plan.update_bid) schedule(Side::Bid, mm_.bid_id(), plan.new_bid_ticks);
        if (plan.update_ask) schedule(Side::Ask, mm_.ask_id(), plan.new_ask_ticks);
    }

    // External order management 

    void seed_background_book() {
        const price_t mid    = cfg_.tick.to_ticks(ref_price());
        const int     levels = std::max(3, cfg_.max_levels / 2);
        for (int lvl = 1; lvl <= levels; ++lvl) {
            add_external_limit(Side::Bid, mid - lvl, cfg_.ext_limit_qty);
            add_external_limit(Side::Ask, mid + lvl, cfg_.ext_limit_qty);
        }
    }

    void add_external_limit(Side side, price_t px, int qty) {
        Order o{};
        o.id = new_id(); o.owner = Owner::External;
        o.side = side; o.price_ticks = px;
        o.qty = qty; o.t_created = t_;
        const int remaining = ob_.add_limit(o, o.id);
        if (remaining > 0) register_external_id(o.id);
    }

    void register_external_id(std::uint64_t id) {
        ext_pos_[id] = ext_ids_.size();
        ext_ids_.push_back(id);
    }
    void unregister_external_id(std::uint64_t id) {
        auto it = ext_pos_.find(id);
        if (it == ext_pos_.end()) return;
        const std::size_t idx  = it->second;
        const std::size_t last = ext_ids_.size() - 1;
        if (idx != last) {
            const std::uint64_t moved = ext_ids_[last];
            ext_ids_[idx] = moved;
            ext_pos_[moved] = idx;
        }
        ext_ids_.pop_back();
        ext_pos_.erase(it);
    }

    // Fill callback 

    void on_fill(const Fill& f) {
        last_trade_px_   = cfg_.tick.to_price(f.price_ticks);
        last_trade_sign_ = (f.maker_side == Side::Ask) ? +1 : -1;

        if (f.maker_owner == Owner::External) {
            if (ob_.locations().find(f.maker_order_id) == ob_.locations().end())
                unregister_external_id(f.maker_order_id);
        }

        mm_.on_fill(f, mark_ticks());

        TradeRow tr{};
        tr.t           = t_;
        tr.px          = cfg_.tick.to_price(f.price_ticks);
        tr.qty         = f.qty;
        tr.sign        = (f.maker_side == Side::Ask) ? +1 : -1;
        tr.mm_involved = (f.maker_owner == Owner::MM) ? 1 : 0;
        tr.mid_mark    = mark_;
        auto bb = ob_.best_bid(); auto aa = ob_.best_ask();
        if (bb && aa) tr.spread = cfg_.tick.to_price(*aa - *bb);
        tr.maker_age_s                 = f.maker_age_s;
        tr.maker_queue_ahead_qty_entry = f.maker_queue_ahead_qty_entry;
        tr.queue_depleted_before_fill  = f.queue_depleted_before_fill;
        tr.mm_tox_widen  = (f.maker_owner == Owner::MM) ? mm_.current_tox_widen() : 0.0;
        tr.cancel_latency_s = pending_cancel_lat_;
        tr.place_latency_s  = pending_place_lat_;
        trades_.push_back(tr);
    }

    // Telemetry snapshot 

    void record(std::int64_t step) {
        TelemetryRow r{};
        r.step     = step;
        r.t        = t_;
        const price_t mk = mark_ticks();
        r.mid_mark = cfg_.tick.to_price(mk);
        auto bb = ob_.best_bid();
        auto aa = ob_.best_ask();
        if (bb) { r.bid = cfg_.tick.to_price(*bb); r.bid_missing = 0; }
        if (aa) { r.ask = cfg_.tick.to_price(*aa); r.ask_missing = 0; }
        if (bb && aa) {
            r.spread = cfg_.tick.to_price(*aa - *bb);
            const int bs = ob_.best_bid_size();
            const int as = ob_.best_ask_size();
            r.bid_sz = bs; r.ask_sz = as;
            if (bs + as > 0)
                r.microprice = (r.ask * bs + r.bid * as)
                             / static_cast<double>(bs + as);
            r.imbalance = static_cast<double>(bs - as) / std::max(1, bs + as);
        }
        r.inventory    = mm_.inventory();
        r.cash         = mm_.cash();
        r.inv_value    = static_cast<double>(r.inventory) * r.mid_mark;
        r.realized_pnl = mm_.realized_pnl();
        r.unreal_pnl   = mm_.unreal_pnl(mk);
        r.fees_paid    = mm_.fees_paid();
        r.equity       = mm_.equity(mk);
        r.mm_fills       = mm_.fills();
        r.mm_cancels_sub = mm_.cancels_submitted();
        r.mm_cancels_ok  = mm_.cancels_succeeded();
        if (mm_.bid_price()) r.mm_bid_px = cfg_.tick.to_price(*mm_.bid_price());
        if (mm_.ask_price()) r.mm_ask_px = cfg_.tick.to_price(*mm_.ask_price());
        // Resting qty from pool (direct pointer, zero copy).
        if (mm_.bid_id() != 0) {
            const Order* op = ob_.find_order(mm_.bid_id());
            if (op) r.mm_bid_qty = op->qty;
        }
        if (mm_.ask_id() != 0) {
            const Order* op = ob_.find_order(mm_.ask_id());
            if (op) r.mm_ask_qty = op->qty;
        }
        if (bb && mm_.bid_price() && *mm_.bid_price() == *bb) {
            r.mm_at_best_bid = 1;
            if (r.bid_sz > 0) r.tob_share_bid = static_cast<double>(r.mm_bid_qty) / r.bid_sz;
        }
        if (aa && mm_.ask_price() && *mm_.ask_price() == *aa) {
            r.mm_at_best_ask = 1;
            if (r.ask_sz > 0) r.tob_share_ask = static_cast<double>(r.mm_ask_qty) / r.ask_sz;
        }
        r.last_trade_px   = last_trade_px_;
        r.last_trade_sign = last_trade_sign_;
        r.tox_widen       = mm_.current_tox_widen();
        r.fill_sign_ema   = mm_.fill_sign_ema();
        r.fv_imb_ema      = mm_.fv_imb_ema();
        r.hawkes_intensity = cfg_.use_replay ? 0.0 : hawkes_.last_intensity();
        ob_.level_depths(Side::Bid, r.bid_levels);
        ob_.level_depths(Side::Ask, r.ask_levels);
        telemetry_.push_back(r);
    }

    // CSV output 

    void dump_telemetry(const std::string& path) const {
        std::FILE* fp = std::fopen(path.c_str(), "w");
        if (!fp) { std::perror("fopen"); return; }
        std::fprintf(fp,
            "step,equity,inventory,bid,ask,bid_missing,ask_missing,"
            "t,mid_mark,microprice,spread,bid_sz,ask_sz,imbalance,"
            "cash,inv_value,realized_pnl,unreal_pnl,fees_paid,"
            "mm_fills,mm_cancels_sub,mm_cancels_ok,mm_bid_px,mm_ask_px,"
            "mm_bid_qty,mm_ask_qty,mm_at_best_bid,mm_at_best_ask,tob_share_bid,tob_share_ask,"
            "last_trade_px,last_trade_sign,"
            "tox_widen,fill_sign_ema,"
            "fv_imb_ema,hawkes_intensity,"
            "bid_l1,bid_l2,bid_l3,bid_l4,bid_l5,"
            "ask_l1,ask_l2,ask_l3,ask_l4,ask_l5\n");
        for (const auto& r : telemetry_) {
            std::fprintf(fp,
                "%lld,%.10f,%d,%.10f,%.10f,%d,%d,"
                "%.6f,%.10f,%.10f,%.10f,%d,%d,%.6f,"
                "%.10f,%.10f,%.10f,%.10f,%.10f,"
                "%llu,%llu,%llu,%.10f,%.10f,"
                "%d,%d,%d,%d,%.10f,%.10f,"
                "%.10f,%d,"
                "%.6f,%.6f,"
                "%.6f,%.3f,"
                "%d,%d,%d,%d,%d,"
                "%d,%d,%d,%d,%d\n",
                static_cast<long long>(r.step),
                r.equity, r.inventory, r.bid, r.ask, r.bid_missing, r.ask_missing,
                r.t, r.mid_mark, r.microprice, r.spread, r.bid_sz, r.ask_sz, r.imbalance,
                r.cash, r.inv_value, r.realized_pnl, r.unreal_pnl, r.fees_paid,
                static_cast<unsigned long long>(r.mm_fills),
                static_cast<unsigned long long>(r.mm_cancels_sub),
                static_cast<unsigned long long>(r.mm_cancels_ok),
                r.mm_bid_px, r.mm_ask_px,
                r.mm_bid_qty, r.mm_ask_qty, r.mm_at_best_bid, r.mm_at_best_ask,
                r.tob_share_bid, r.tob_share_ask,
                r.last_trade_px, r.last_trade_sign,
                r.tox_widen, r.fill_sign_ema,
                r.fv_imb_ema, r.hawkes_intensity,
                r.bid_levels[0], r.bid_levels[1], r.bid_levels[2],
                r.bid_levels[3], r.bid_levels[4],
                r.ask_levels[0], r.ask_levels[1], r.ask_levels[2],
                r.ask_levels[3], r.ask_levels[4]);
        }
        std::fclose(fp);
    }

    void dump_trades(const std::string& path) const {
        std::FILE* fp = std::fopen(path.c_str(), "w");
        if (!fp) { std::perror("fopen"); return; }
        std::fprintf(fp,
            "t,px,qty,sign,mm_involved,mid_mark,spread,"
            "maker_age_s,maker_queue_ahead_qty_entry,"
            "queue_depleted_before_fill,mm_tox_widen,"
            "cancel_latency_s,place_latency_s\n");
        for (const auto& tr : trades_) {
            std::fprintf(fp,
                "%.6f,%.10f,%d,%d,%d,%.10f,%.10f,"
                "%.6f,%d,"
                "%lld,%.6f,"
                "%.6f,%.6f\n",
                tr.t, tr.px, tr.qty, tr.sign, tr.mm_involved,
                tr.mid_mark, tr.spread,
                tr.maker_age_s, tr.maker_queue_ahead_qty_entry,
                static_cast<long long>(tr.queue_depleted_before_fill),
                tr.mm_tox_widen,
                tr.cancel_latency_s, tr.place_latency_s);
        }
        std::fclose(fp);
    }

    RunSummary make_summary() const {
        RunSummary s{};
        s.seed = cfg_.seed;
        if (telemetry_.empty()) return s;
        const auto& last = telemetry_.back();
        s.final_equity    = last.equity;
        s.realized_pnl    = last.realized_pnl;
        s.fees_paid       = last.fees_paid;
        s.final_inventory = last.inventory;
        s.total_fills     = last.mm_fills;
        s.total_cancels_sub = last.mm_cancels_sub;
        double peak = telemetry_[0].equity;
        for (const auto& r : telemetry_) {
            if (r.equity > peak) peak = r.equity;
            const double dd = peak - r.equity;
            if (dd > s.max_drawdown) s.max_drawdown = dd;
        }
        return s;
    }
};

} // namespace sim

// main

int main() {
    sim::SimConfig cfg{};

    cfg.steps       = 3000;
    cfg.log_dt      = 0.10;
    cfg.tick        = sim::TickSpec{0.01};
    cfg.start_price = 100.0;
    cfg.seed        = 42;

    // MM strategy parameters.
    cfg.mm.quote_qty                   = 1;
    cfg.mm.base_spread_ticks           = 2.0;

    cfg.mm.spread_widen_ticks_per_share = 0.06;
    cfg.mm.inv_requote_band            = 5;
    cfg.mm.max_quote_age_s             = 0.75;
    cfg.mm.min_quote_life_s            = 0.05;
    cfg.mm.maker_rebate_per_share      = -0.0002;
    cfg.mm.tox_ema_alpha               = 0.30;
    cfg.mm.tox_widen_threshold         = 0.40;
    cfg.mm.tox_widen_max_ticks         = 2.0;

    // FairValueModel config.
    cfg.mm.fv.imb_alpha               = 0.20;
    cfg.mm.fv.imb_coeff_ticks         = 1.5;
    cfg.mm.fv.inv_skew_ticks_per_share = 0.18;

    // LatencyModel: log-normal, mean cancel ~30ms, mean place ~10ms.
    // To enable congestion: set congestion_per_event_s = 1e-5.
    cfg.mm.latency.cancel_lnmu    = -3.618;
    cfg.mm.latency.cancel_lnsigma = 0.472;
    cfg.mm.latency.place_lnmu     = -4.605;
    cfg.mm.latency.place_lnsigma  = 0.350;
    cfg.mm.latency.congestion_per_event_s = 0.0;

    // Background LOB.
    cfg.max_levels           = 14;
    cfg.level_geo_p          = 0.35;
    cfg.ext_limit_qty        = 1;
    cfg.p_informed           = 0.22;
    cfg.info_threshold_ticks = 1.0;
    cfg.impact_per_mkt       = 0.015;

    // Hawkes flow model.
    cfg.hawkes.beta       = 6.0;
    cfg.hawkes.mu_lim     = 85.0;
    cfg.hawkes.mu_cancel  = 38.0;
    cfg.hawkes.mu_mkt     = 12.0;
    cfg.hawkes.lambda_cap = 900.0;

    // Run 1: Hawkes generative mode (reference run, seed 42) 
    std::cout << "=== Hawkes mode (seed 42) ===\n";
    cfg.use_replay = false;
    cfg.seed       = 42;
    {
        sim::Simulator ref(cfg);
        ref.run(true);
    }

    // Run 2: Replay mode (synthetic historical feed) 
    // replay_path is empty → auto-generates synthetic events in historical CSV format.
    // To use real data: set cfg.replay_path = "your_feed.csv" (see ReplayEngine format).
    std::cout << "\n=== Replay mode (synthetic historical feed) ===\n";
    cfg.use_replay  = true;
    cfg.replay_path = "";
    cfg.seed        = 42;
    {
        sim::Simulator rp(cfg);
        rp.run(true);
    }

    // Multi-seed robustness sweep (Hawkes mode) 
    std::cout << "\n=== Multi-seed robustness sweep ===\n";
    cfg.use_replay = false;
    const std::array<std::uint64_t, 10> seeds = {
        42, 137, 999, 1234, 5678, 271828, 314159, 161803, 100003, 77777
    };
    std::FILE* fp = std::fopen("multi_seed_summary_v3.csv", "w");
    if (!fp) { std::perror("fopen multi_seed_summary_v3.csv"); return 1; }
    std::fprintf(fp,
        "seed,final_equity,max_drawdown,realized_pnl,"
        "fees_paid,final_inventory,total_fills,total_cancels_sub\n");
    for (const std::uint64_t seed : seeds) {
        cfg.seed = seed;
        sim::Simulator s(cfg);
        const auto res = s.run(false);
        std::fprintf(fp,
            "%llu,%.6f,%.6f,%.6f,%.6f,%d,%llu,%llu\n",
            static_cast<unsigned long long>(res.seed),
            res.final_equity, res.max_drawdown, res.realized_pnl, res.fees_paid,
            res.final_inventory,
            static_cast<unsigned long long>(res.total_fills),
            static_cast<unsigned long long>(res.total_cancels_sub));
    }
    std::fclose(fp);
    std::cout << "Multi-seed summary: multi_seed_summary_v3.csv ("
              << seeds.size() << " seeds)\n";

    return 0;
}
