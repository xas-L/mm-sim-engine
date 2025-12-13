#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <list>
#include <map>
#include <numeric>
#include <optional>
#include <queue>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace sim {


// helpers


inline bool is_finite(double x) noexcept { return std::isfinite(x); }

using price_t = std::int64_t; // integer ticks

struct TickSpec {
  double tick_size{0.01};

  price_t to_ticks(double px) const noexcept {
    return static_cast<price_t>(std::llround(px / tick_size));
  }

  double to_price(price_t ticks) const noexcept {
    return static_cast<double>(ticks) * tick_size;
  }
};

enum class Side : std::uint8_t { Bid = 0, Ask = 1 };
enum class Owner : std::uint8_t { External = 0, MM = 1 };

inline Side opposite(Side s) noexcept { return (s == Side::Bid) ? Side::Ask : Side::Bid; }

struct Order {
  std::uint64_t id{0};
  Owner owner{Owner::External};
  Side side{Side::Bid};
  price_t price_ticks{0};
  int qty{0};                // remaining qty
  double t_created{0.0};     // seconds
  int queue_ahead_qty{0};    // snapshot when inserted (useful for MM telemetry)
};

struct Fill {
  double t{0.0};
  price_t price_ticks{0};
  int qty{0};

  Owner maker_owner{Owner::External};
  Owner taker_owner{Owner::External};

  Side maker_side{Side::Bid}; // side of resting order
  std::uint64_t maker_order_id{0};
  std::uint64_t taker_order_id{0};

  // Extra microstructure diagnostics
  double maker_age_s{0.0};
  int maker_queue_ahead_qty_entry{0};
};


// OrderBook (price-time priority, unit-size friendly)


struct OrderHandle {
  Side side{Side::Bid};
  price_t price_ticks{0};
  std::list<Order>::iterator it{};
};

class OrderBook {
public:
  using FillCallback = std::function<void(const Fill&)>;

  explicit OrderBook(TickSpec tick) : tick_(tick) {}

  void set_fill_callback(FillCallback cb) { on_fill_ = std::move(cb); }

  std::optional<price_t> best_bid() const noexcept {
    if (bids_.empty()) return std::nullopt;
    return bids_.begin()->first;
  }

  std::optional<price_t> best_ask() const noexcept {
    if (asks_.empty()) return std::nullopt;
    return asks_.begin()->first;
  }

  // Top-of-book queue sizes (at best price)
  int best_bid_size() const noexcept {
    if (bids_.empty()) return 0;
    return level_size(bids_.begin()->second);
  }

  int best_ask_size() const noexcept {
    if (asks_.empty()) return 0;
    return level_size(asks_.begin()->second);
  }

  // Total depth (rough imbalance proxy, capped at a few levels)
  int depth_size(Side side, int levels) const noexcept {
    int tot = 0;
    if (levels <= 0) return 0;

    if (side == Side::Bid) {
      int k = 0;
      for (const auto& [px, lvl] : bids_) {
        tot += level_size(lvl);
        if (++k >= levels) break;
      }
    } else {
      int k = 0;
      for (const auto& [px, lvl] : asks_) {
        tot += level_size(lvl);
        if (++k >= levels) break;
      }
    }
    return tot;
  }

  bool cancel(std::uint64_t order_id) {
    auto it = loc_.find(order_id);
    if (it == loc_.end()) return false;

    const OrderHandle h = it->second;

    if (h.side == Side::Bid) {
      auto lvl_it = bids_.find(h.price_ticks);
      if (lvl_it == bids_.end()) {
        loc_.erase(it);
        return false;
      }
      lvl_it->second.erase(h.it);
      loc_.erase(it);
      if (lvl_it->second.empty()) bids_.erase(lvl_it);
      return true;
    }

    // Ask side
    auto lvl_it = asks_.find(h.price_ticks);
    if (lvl_it == asks_.end()) {
      loc_.erase(it);
      return false;
    }
    lvl_it->second.erase(h.it);
    loc_.erase(it);
    if (lvl_it->second.empty()) asks_.erase(lvl_it);
    return true;
  }

  // Adds a limit order. If marketable, it executes immediately up to the limit price.
  // Returns remaining qty that ended up resting (0 if fully executed).
  int add_limit(Order o, std::uint64_t taker_order_id_for_cross = 0) {
    if (o.qty <= 0) return 0;

    // If crosses, treat as a taker up to o.price_ticks.
    if (o.side == Side::Bid) {
      match_against_asks(o, /*limit_price=*/o.price_ticks, taker_order_id_for_cross, /*taker_owner=*/o.owner);
    } else {
      match_against_bids(o, /*limit_price=*/o.price_ticks, taker_order_id_for_cross, /*taker_owner=*/o.owner);
    }

    if (o.qty <= 0) return 0;

    // Rest the remainder.
    if (o.side == Side::Bid) {
      auto& lvl = bids_[o.price_ticks];
      o.queue_ahead_qty = level_size(lvl);
      lvl.push_back(o);
      auto it = std::prev(lvl.end());
      loc_[o.id] = OrderHandle{o.side, o.price_ticks, it};
      return o.qty;
    }

    auto& lvl = asks_[o.price_ticks];
    o.queue_ahead_qty = level_size(lvl);
    lvl.push_back(o);
    auto it = std::prev(lvl.end());
    loc_[o.id] = OrderHandle{o.side, o.price_ticks, it};
    return o.qty;
  }

  // Market order consumes opposite side, no price limit.
  void add_market(std::uint64_t taker_id, Owner taker_owner, Side taker_side, int qty, double t) {
    if (qty <= 0) return;

    Order stub{};
    stub.id = taker_id;
    stub.owner = taker_owner;
    stub.side = taker_side;
    stub.qty = qty;
    stub.t_created = t;

    if (taker_side == Side::Bid) {
      match_against_asks(stub, /*limit_price=*/std::nullopt, taker_id, taker_owner);
    } else {
      match_against_bids(stub, /*limit_price=*/std::nullopt, taker_id, taker_owner);
    }
  }

  // Useful for sampling cancels / telemetry
  const std::unordered_map<std::uint64_t, OrderHandle>& locations() const noexcept { return loc_; }

private:
  using Level = std::list<Order>;
  using BidBook = std::map<price_t, Level, std::greater<price_t>>; // high -> low
  using AskBook = std::map<price_t, Level, std::less<price_t>>;    // low -> high

  TickSpec tick_{};
  BidBook bids_{};
  AskBook asks_{};

  std::unordered_map<std::uint64_t, OrderHandle> loc_{};
  FillCallback on_fill_{};

  static int level_size(const Level& lvl) noexcept {
    int tot = 0;
    for (const auto& o : lvl) tot += o.qty;
    return tot;
  }

  void emit_fill(const Fill& f) { if (on_fill_) on_fill_(f); }

  void match_against_asks(Order& taker, std::optional<price_t> limit_price,
                          std::uint64_t taker_order_id, Owner taker_owner) {
    while (taker.qty > 0 && !asks_.empty()) {
      auto best_it = asks_.begin();
      const price_t ask_px = best_it->first;
      if (limit_price && ask_px > *limit_price) break;

      auto& lvl = best_it->second;
      auto it = lvl.begin();
      while (it != lvl.end() && taker.qty > 0) {
        Order& maker = *it;
        const int fill_qty = std::min(taker.qty, maker.qty);

        Fill f{};
        f.t = taker.t_created;
        f.price_ticks = ask_px;
        f.qty = fill_qty;
        f.maker_owner = maker.owner;
        f.taker_owner = taker_owner;
        f.maker_side = maker.side;
        f.maker_order_id = maker.id;
        f.taker_order_id = taker_order_id;
        f.maker_age_s = taker.t_created - maker.t_created;
        f.maker_queue_ahead_qty_entry = maker.queue_ahead_qty;
        emit_fill(f);

        taker.qty -= fill_qty;
        maker.qty -= fill_qty;

        if (maker.qty == 0) {
          loc_.erase(maker.id);
          it = lvl.erase(it);
        } else {
          ++it;
        }
      }
      if (lvl.empty()) asks_.erase(best_it);
    }
  }

  void match_against_bids(Order& taker, std::optional<price_t> limit_price,
                          std::uint64_t taker_order_id, Owner taker_owner) {
    while (taker.qty > 0 && !bids_.empty()) {
      auto best_it = bids_.begin();
      const price_t bid_px = best_it->first;
      if (limit_price && bid_px < *limit_price) break;

      auto& lvl = best_it->second;
      auto it = lvl.begin();
      while (it != lvl.end() && taker.qty > 0) {
        Order& maker = *it;
        const int fill_qty = std::min(taker.qty, maker.qty);

        Fill f{};
        f.t = taker.t_created;
        f.price_ticks = bid_px;
        f.qty = fill_qty;
        f.maker_owner = maker.owner;
        f.taker_owner = taker_owner;
        f.maker_side = maker.side;
        f.maker_order_id = maker.id;
        f.taker_order_id = taker_order_id;
        f.maker_age_s = taker.t_created - maker.t_created;
        f.maker_queue_ahead_qty_entry = maker.queue_ahead_qty;
        emit_fill(f);

        taker.qty -= fill_qty;
        maker.qty -= fill_qty;

        if (maker.qty == 0) {
          loc_.erase(maker.id);
          it = lvl.erase(it);
        } else {
          ++it;
        }
      }
      if (lvl.empty()) bids_.erase(best_it);
    }
  }
};


// Market maker (accounting + quoting logic)


struct MMConfig {
  int quote_qty{1};

  // Quote model (all in ticks)
  double base_spread_ticks{2.0};            // minimum spread in ticks
  double inv_skew_ticks_per_share{0.15};    // reservation shift per share of inventory
  double spread_widen_ticks_per_share{0.05}; // widen spread as inv grows

  int inv_requote_band{5};
  double max_quote_age_s{0.75};
  double min_quote_life_s{0.05}; // avoid cancel spam

  // Latency model
  double cancel_latency_s{0.03};
  double place_latency_s{0.01};

  // Fees (sign convention: positive = fee paid, negative = rebate received)
  double maker_rebate_per_share{-0.0002}; // negative means we receive money per share
};

class MarketMaker {
public:
  MarketMaker(OrderBook& ob, TickSpec tick, MMConfig cfg)
    : ob_(ob), tick_(tick), cfg_(cfg) {}

  void on_fill(const Fill& f, price_t mark_ticks) {
    if (f.maker_owner != Owner::MM) return;

    ++mm_fills_;
    mm_volume_ += f.qty;

    const double px = tick_.to_price(f.price_ticks);
    const double fee = static_cast<double>(f.qty) * cfg_.maker_rebate_per_share;
    fees_paid_ += fee;

    if (f.maker_side == Side::Bid) {
      apply_trade(/*is_buy=*/true, px, f.qty, fee);
      if (f.maker_order_id == mm_bid_id_) mm_bid_id_ = 0;
    } else {
      apply_trade(/*is_buy=*/false, px, f.qty, fee);
      if (f.maker_order_id == mm_ask_id_) mm_ask_id_ = 0;
    }

    last_mark_ticks_ = mark_ticks;
  }

  int inventory() const noexcept { return inv_; }
  double cash() const noexcept { return cash_; }
  double realized_pnl() const noexcept { return realized_pnl_; }
  double fees_paid() const noexcept { return fees_paid_; }
  std::uint64_t fills() const noexcept { return mm_fills_; }
  std::uint64_t cancels_submitted() const noexcept { return cancels_submitted_; }
  std::uint64_t cancels_succeeded() const noexcept { return cancels_succeeded_; }

  double equity(price_t mark_ticks) const noexcept {
    return cash_ + static_cast<double>(inv_) * tick_.to_price(mark_ticks);
  }

  double unreal_pnl(price_t mark_ticks) const noexcept {
    if (inv_ == 0) return 0.0;
    const double mark = tick_.to_price(mark_ticks);
    if (inv_ > 0) return (mark - avg_cost_) * static_cast<double>(inv_);
    return (avg_cost_ - mark) * static_cast<double>(-inv_);
  }

  std::uint64_t bid_id() const noexcept { return mm_bid_id_; }
  std::uint64_t ask_id() const noexcept { return mm_ask_id_; }

  std::optional<price_t> bid_price() const noexcept { return bid_px_; }
  std::optional<price_t> ask_price() const noexcept { return ask_px_; }

  void note_quote_placed(Side side, std::uint64_t id, price_t px, double t_now) {
    if (side == Side::Bid) {
      mm_bid_id_ = id;
      bid_px_ = px;
      bid_t_placed_ = t_now;
    } else {
      mm_ask_id_ = id;
      ask_px_ = px;
      ask_t_placed_ = t_now;
    }
    last_quote_update_t_ = t_now;
    last_quote_mid_ticks_ = last_mid_for_quote_;
  }

  void note_quote_canceled(Side side, std::uint64_t id, bool success) {
    ++cancels_submitted_;
    if (success) ++cancels_succeeded_;

    if (side == Side::Bid && id == mm_bid_id_) {
      mm_bid_id_ = 0;
      bid_px_.reset();
    }
    if (side == Side::Ask && id == mm_ask_id_) {
      mm_ask_id_ = 0;
      ask_px_.reset();
    }
  }

  struct QuotePlan {
    bool update_bid{false};
    bool update_ask{false};
    price_t new_bid_ticks{0};
    price_t new_ask_ticks{0};
  };

  QuotePlan plan_quotes(double t_now, price_t mark_mid_ticks, std::optional<price_t> bb, std::optional<price_t> aa) {
    last_mid_for_quote_ = mark_mid_ticks;

    const double inv = static_cast<double>(inv_);
    const double reservation = static_cast<double>(mark_mid_ticks) - cfg_.inv_skew_ticks_per_share * inv;

    const double spread_ticks = std::max(1.0, cfg_.base_spread_ticks + cfg_.spread_widen_ticks_per_share * std::abs(inv));
    const double half = 0.5 * spread_ticks;

    price_t bid_ticks = static_cast<price_t>(std::floor(reservation - half));
    price_t ask_ticks = static_cast<price_t>(std::ceil(reservation + half));
    if (ask_ticks <= bid_ticks) ask_ticks = bid_ticks + 1;

    // Keep MM passive (do not cross)
    if (aa && bid_ticks >= *aa) bid_ticks = *aa - 1;
    if (bb && ask_ticks <= *bb) ask_ticks = *bb + 1;
    if (ask_ticks <= bid_ticks) ask_ticks = bid_ticks + 1;

    QuotePlan p{};
    p.new_bid_ticks = bid_ticks;
    p.new_ask_ticks = ask_ticks;

    const bool mid_moved = (!has_last_quote_mid_) || (std::llabs(mark_mid_ticks - last_quote_mid_ticks_) >= 1);
    const bool inv_band = (cfg_.inv_requote_band > 0)
      && ((std::abs(inv_) / cfg_.inv_requote_band) != (std::abs(last_inv_for_quote_) / cfg_.inv_requote_band));
    const bool stale_bid = bid_px_ && ((t_now - bid_t_placed_) > cfg_.max_quote_age_s);
    const bool stale_ask = ask_px_ && ((t_now - ask_t_placed_) > cfg_.max_quote_age_s);

    const bool too_soon = (t_now - last_quote_update_t_) < cfg_.min_quote_life_s;

    if (!too_soon && (mid_moved || inv_band || stale_bid)) {
      if (!bid_px_ || *bid_px_ != bid_ticks) p.update_bid = true;
    }
    if (!too_soon && (mid_moved || inv_band || stale_ask)) {
      if (!ask_px_ || *ask_px_ != ask_ticks) p.update_ask = true;
    }

    last_inv_for_quote_ = inv_;
    has_last_quote_mid_ = true;
    return p;
  }

private:
  OrderBook& ob_;
  TickSpec tick_;
  MMConfig cfg_{};

  // Position/accounting
  int inv_{0};
  double cash_{0.0};
  double avg_cost_{0.0};      // avg entry cost for current inventory magnitude
  double realized_pnl_{0.0};
  double fees_paid_{0.0};     // negative = net rebates

  // Quote state
  std::uint64_t mm_bid_id_{0};
  std::uint64_t mm_ask_id_{0};
  std::optional<price_t> bid_px_{};
  std::optional<price_t> ask_px_{};
  double bid_t_placed_{-1e9};
  double ask_t_placed_{-1e9};

  double last_quote_update_t_{-1e9};
  price_t last_quote_mid_ticks_{0};
  bool has_last_quote_mid_{false};
  int last_inv_for_quote_{0};
  price_t last_mid_for_quote_{0};

  price_t last_mark_ticks_{0};

  // Stats
  std::uint64_t mm_fills_{0};
  std::uint64_t mm_volume_{0};
  std::uint64_t cancels_submitted_{0};
  std::uint64_t cancels_succeeded_{0};

  void apply_trade(bool is_buy, double px, int qty, double fee_cash) {
    // fee_cash: negative = rebate, positive = fee paid
    if (is_buy) {
      cash_ -= px * static_cast<double>(qty);
      cash_ -= fee_cash;

      if (inv_ >= 0) {
        const int new_inv = inv_ + qty;
        const double new_cost =
          (avg_cost_ * static_cast<double>(inv_) + px * static_cast<double>(qty))
          / static_cast<double>(std::max(1, new_inv));
        inv_ = new_inv;
        avg_cost_ = new_cost;
      } else {
        const int cover = std::min(qty, -inv_);
        realized_pnl_ += (avg_cost_ - px) * static_cast<double>(cover);
        inv_ += cover;

        const int remaining = qty - cover;
        if (remaining > 0) {
          inv_ = remaining;
          avg_cost_ = px;
        } else if (inv_ == 0) {
          avg_cost_ = 0.0;
        }
      }
    } else {
      cash_ += px * static_cast<double>(qty);
      cash_ -= fee_cash;

      if (inv_ <= 0) {
        const int new_inv = inv_ - qty;
        const int abs_old = -inv_;
        const int abs_new = -new_inv;
        const double new_cost =
          (avg_cost_ * static_cast<double>(abs_old) + px * static_cast<double>(qty))
          / static_cast<double>(std::max(1, abs_new));
        inv_ = new_inv;
        avg_cost_ = new_cost;
      } else {
        const int close = std::min(qty, inv_);
        realized_pnl_ += (px - avg_cost_) * static_cast<double>(close);
        inv_ -= close;

        const int remaining = qty - close;
        if (remaining > 0) {
          inv_ = -remaining;
          avg_cost_ = px;
        } else if (inv_ == 0) {
          avg_cost_ = 0.0;
        }
      }
    }
  }
};


// Hawkes-style clustered event flow (6 event types)


enum class EType : std::uint8_t {
  LimAddBid = 0,
  LimAddAsk = 1,
  CancelBid = 2,
  CancelAsk = 3,
  MktBuy = 4,
  MktSell = 5,
  Count = 6
};

struct HawkesConfig {
  double beta{6.0};

  // Baseline intensities (events per second). State-adjusted.
  double mu_lim{80.0};
  double mu_cancel{35.0};
  double mu_mkt{12.0};

  // Excitation strength.
  double alpha_self{0.7};
  double alpha_cross{0.15};

  double lambda_cap{800.0};
};

class Hawkes6 {
public:
  explicit Hawkes6(HawkesConfig cfg, std::mt19937_64& rng) : cfg_(cfg), rng_(rng) {
    const int K = static_cast<int>(EType::Count);
    S_.assign(K, 0.0);
    mu_.assign(K, 0.0);
    lambda_.assign(K, 0.0);
  }

  void set_state(double spread_ticks, double imbalance, double info_edge_ticks) {
    const double spread_boost = 1.0 + 0.03 * std::max(0.0, spread_ticks - 2.0);
    const double lim = cfg_.mu_lim * spread_boost;

    const double cancel_bid = cfg_.mu_cancel * (1.0 + 0.8 * std::max(0.0, +imbalance));
    const double cancel_ask = cfg_.mu_cancel * (1.0 + 0.8 * std::max(0.0, -imbalance));

    const double z = std::clamp(info_edge_ticks / 2.0, -6.0, 6.0);
    const double buy_tilt = 1.0 / (1.0 + std::exp(-z));
    const double sell_tilt = 1.0 - buy_tilt;

    const double mkt_buy = cfg_.mu_mkt * (0.35 + 1.3 * buy_tilt);
    const double mkt_sell = cfg_.mu_mkt * (0.35 + 1.3 * sell_tilt);

    mu_[static_cast<int>(EType::LimAddBid)] = lim;
    mu_[static_cast<int>(EType::LimAddAsk)] = lim;
    mu_[static_cast<int>(EType::CancelBid)] = cancel_bid;
    mu_[static_cast<int>(EType::CancelAsk)] = cancel_ask;
    mu_[static_cast<int>(EType::MktBuy)] = mkt_buy;
    mu_[static_cast<int>(EType::MktSell)] = mkt_sell;
  }

  // Sample next external event time/type via Ogata thinning.
  std::pair<double, EType> next(double t_now) {
    if (!has_t_) {
      t_ = t_now;
      has_t_ = true;
    } else if (t_now > t_) {
      decay(t_now - t_);
      t_ = t_now;
    }

    std::exponential_distribution<double> expd(1.0);
    std::uniform_real_distribution<double> U(0.0, 1.0);

    for (;;) {
      update_lambda();
      double lambda_tot = std::accumulate(lambda_.begin(), lambda_.end(), 0.0);
      lambda_tot = std::min(lambda_tot, cfg_.lambda_cap);
      if (lambda_tot <= 0.0) lambda_tot = 1e-9;

      const double M = lambda_tot * 1.10;
      const double dt = expd(rng_) / M;

      decay(dt);
      t_ += dt;

      update_lambda();
      const double lambda_new = std::min(std::accumulate(lambda_.begin(), lambda_.end(), 0.0), cfg_.lambda_cap);

      if (U(rng_) * M <= lambda_new) {
        const double r = U(rng_) * lambda_new;
        double c = 0.0;
        for (int k = 0; k < static_cast<int>(EType::Count); ++k) {
          c += lambda_[k];
          if (r <= c) {
            const auto et = static_cast<EType>(k);
            S_[k] += 1.0;
            return {t_, et};
          }
        }
        S_.back() += 1.0;
        return {t_, EType::MktSell};
      }
      // reject and loop
    }
  }

private:
  HawkesConfig cfg_{};
  std::mt19937_64& rng_;

  bool has_t_{false};
  double t_{0.0};

  std::vector<double> S_;
  std::vector<double> mu_;
  std::vector<double> lambda_;

  void decay(double dt) {
    if (dt <= 0.0) return;
    const double f = std::exp(-cfg_.beta * dt);
    for (double& x : S_) x *= f;
  }

  void update_lambda() {
    const double sumS = std::accumulate(S_.begin(), S_.end(), 0.0);
    for (int k = 0; k < static_cast<int>(EType::Count); ++k) {
      const double self = S_[k];
      const double cross = sumS - self;
      lambda_[k] = mu_[k] + cfg_.alpha_self * self + cfg_.alpha_cross * cross;
      if (lambda_[k] < 0.0) lambda_[k] = 0.0;
    }
  }
};


// Action queue for MM latency


enum class ActionType : std::uint8_t { Cancel = 0, Place = 1, RequoteCheck = 2 };

struct Action {
  double t{0.0};
  std::uint64_t seq{0};
  ActionType type{ActionType::RequoteCheck};

  // id for cancel/place
  std::uint64_t order_id{0};

  // Cancel
  Side side{Side::Bid};

  // Place
  price_t price_ticks{0};
  int qty{0};
  Owner owner{Owner::MM};

  bool operator>(const Action& other) const noexcept {
    if (t != other.t) return t > other.t;
    return seq > other.seq;
  }
};


// Simulator


struct SimConfig {
  int steps{3000};
  double log_dt{0.10};

  std::uint64_t seed{42};

  TickSpec tick{0.01};
  int lot_size{1};

  // Fundamental / reference price model
  double start_price{100.0};
  double ou_kappa{1.5};
  double ou_theta{100.0};
  double ou_sigma{0.20};

  // Micro price impact state
  double impact_kappa{8.0};
  double impact_sigma{0.0};
  double impact_per_mkt{0.015};

  // External limit placement
  int max_levels{12};
  double level_geo_p{0.35};
  int ext_limit_qty{1};

  // Cancel sampling
  int cancel_sample_tries{12};
  double min_ext_order_age_s{0.03};

  // Informed vs uninformed
  double p_informed{0.22};
  double info_threshold_ticks{1.0};

  // MM
  MMConfig mm{};

  HawkesConfig hawkes{};
};

struct TelemetryRow {
  std::int64_t step{0};
  double t{0.0};

  // Legacy columns
  double equity{0.0};
  int inventory{0};
  double bid{std::numeric_limits<double>::quiet_NaN()};
  double ask{std::numeric_limits<double>::quiet_NaN()};
  int bid_missing{1};
  int ask_missing{1};

  // Extra columns
  double mid_mark{0.0};
  double microprice{std::numeric_limits<double>::quiet_NaN()};
  double spread{std::numeric_limits<double>::quiet_NaN()};
  int bid_sz{0};
  int ask_sz{0};
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

  int mm_bid_qty{0};
  int mm_ask_qty{0};
  int mm_at_best_bid{0};
  int mm_at_best_ask{0};
  double tob_share_bid{std::numeric_limits<double>::quiet_NaN()};
  double tob_share_ask{std::numeric_limits<double>::quiet_NaN()};

  double last_trade_px{std::numeric_limits<double>::quiet_NaN()};
  int last_trade_sign{0}; // +1 buy, -1 sell
};

struct TradeRow {
  double t{0.0};
  double px{0.0};
  int qty{0};
  int sign{0}; // +1 buyer-initiated (taker buy), -1 seller-initiated
  int mm_involved{0}; // 1 if MM was maker
  double mid_mark{0.0};
  double spread{std::numeric_limits<double>::quiet_NaN()};
  double maker_age_s{0.0};
  int maker_queue_ahead_qty_entry{0};
};

class Simulator {
public:
  explicit Simulator(SimConfig cfg)
    : cfg_(cfg),
      rng_(cfg.seed),
      ob_(cfg.tick),
      mm_(ob_, cfg.tick, cfg.mm),
      hawkes_(cfg.hawkes, rng_) {

    ob_.set_fill_callback([this](const Fill& f) { this->on_fill(f); });

    t_ = 0.0;
    mark_ = cfg_.start_price;
    impact_ = 0.0;
    seed_background_book();
  }

  void run() {
    const int N = cfg_.steps;

    double next_log = cfg_.log_dt;
    std::int64_t step = 0;

    push_action({/*t=*/0.0, /*seq=*/next_seq_++, ActionType::RequoteCheck});

    update_hawkes_state();
    auto [t_ext, et] = hawkes_.next(t_);
    next_ext_t_ = t_ext;
    next_ext_type_ = et;

    while (step < N) {
      const double t_next_action = next_action_time();
      const double t_next = std::min({next_log, t_next_action, next_ext_t_});

      advance_time(t_next);

      if (t_next == next_log) {
        record(step);
        ++step;
        next_log += cfg_.log_dt;
        continue;
      }

      if (t_next == t_next_action) {
        do_action(pop_action());
        continue;
      }

      do_external_event(next_ext_type_);
      update_hawkes_state();
      std::tie(next_ext_t_, next_ext_type_) = hawkes_.next(t_);
    }

    if (static_cast<int>(telemetry_.size()) < N) record(step);

    dump_csv("telemetry_v5.csv");
    dump_trades_csv("trades_v5.csv");
    std::cout << "CSV written: telemetry_v5.csv\n";
    std::cout << "Trades written: trades_v5.csv\n";
  }

private:
  SimConfig cfg_{};
  std::mt19937_64 rng_;
  OrderBook ob_;
  MarketMaker mm_;
  Hawkes6 hawkes_;

  double t_{0.0};
  double mark_{0.0};
  double impact_{0.0};

  std::vector<TelemetryRow> telemetry_{};
  std::vector<TradeRow> trades_{};

  double last_trade_px_{std::numeric_limits<double>::quiet_NaN()};
  int last_trade_sign_{0};

  double next_ext_t_{0.0};
  EType next_ext_type_{EType::LimAddBid};

  std::priority_queue<Action, std::vector<Action>, std::greater<Action>> actions_{};
  std::uint64_t next_seq_{1};

  std::uint64_t next_order_id_{1};

  // Active external order ids for O(1) random cancel sampling
  std::vector<std::uint64_t> ext_ids_{};
  std::unordered_map<std::uint64_t, std::size_t> ext_pos_{};

  std::uint64_t new_id() { return next_order_id_++; }

  double ref_price() const noexcept { return mark_ + impact_; }
  price_t mark_ticks() const noexcept { return cfg_.tick.to_ticks(mark_); }
  price_t ref_ticks() const noexcept { return cfg_.tick.to_ticks(ref_price()); }

  double next_action_time() const noexcept {
    if (actions_.empty()) return std::numeric_limits<double>::infinity();
    return actions_.top().t;
  }

  void push_action(Action a) { actions_.push(std::move(a)); }

  Action pop_action() {
    Action a = actions_.top();
    actions_.pop();
    return a;
  }

  void seed_background_book() {
    const price_t mid = cfg_.tick.to_ticks(ref_price());
    const int levels = std::max(3, cfg_.max_levels / 2);

    for (int lvl = 1; lvl <= levels; ++lvl) {
      add_external_limit(Side::Bid, mid - lvl, cfg_.ext_limit_qty);
      add_external_limit(Side::Ask, mid + lvl, cfg_.ext_limit_qty);
    }
  }

  void add_external_limit(Side side, price_t px, int qty) {
    Order o{};
    o.id = new_id();
    o.owner = Owner::External;
    o.side = side;
    o.price_ticks = px;
    o.qty = qty;
    o.t_created = t_;

    const int remaining = ob_.add_limit(o, /*taker_order_id_for_cross=*/o.id);
    if (remaining > 0) register_external_id(o.id);
  }

  void register_external_id(std::uint64_t id) {
    ext_pos_[id] = ext_ids_.size();
    ext_ids_.push_back(id);
  }

  void unregister_external_id(std::uint64_t id) {
    auto it = ext_pos_.find(id);
    if (it == ext_pos_.end()) return;

    const std::size_t idx = it->second;
    const std::size_t last = ext_ids_.size() - 1;

    if (idx != last) {
      const std::uint64_t moved = ext_ids_[last];
      ext_ids_[idx] = moved;
      ext_pos_[moved] = idx;
    }
    ext_ids_.pop_back();
    ext_pos_.erase(it);
  }

  void on_fill(const Fill& f) {
    last_trade_px_ = cfg_.tick.to_price(f.price_ticks);
    last_trade_sign_ = (f.maker_side == Side::Ask) ? +1 : -1; // taker bought if maker was ask

    // If maker was external and got fully consumed, remove from external id pool
    if (f.maker_owner == Owner::External) {
      const auto& locs = ob_.locations();
      if (locs.find(f.maker_order_id) == locs.end()) {
        unregister_external_id(f.maker_order_id);
      }
    }

    mm_.on_fill(f, mark_ticks());

    // Trade log
    TradeRow tr{};
    tr.t = t_;
    tr.px = cfg_.tick.to_price(f.price_ticks);
    tr.qty = f.qty;
    tr.sign = (f.maker_side == Side::Ask) ? +1 : -1;
    tr.mm_involved = (f.maker_owner == Owner::MM) ? 1 : 0;
    tr.mid_mark = mark_;
    auto bb = ob_.best_bid();
    auto aa = ob_.best_ask();
    if (bb && aa) tr.spread = cfg_.tick.to_price(*aa - *bb);
    tr.maker_age_s = f.maker_age_s;
    tr.maker_queue_ahead_qty_entry = f.maker_queue_ahead_qty_entry;
    trades_.push_back(tr);
  }

  void advance_time(double t_new) {
    if (t_new < t_) return;
    const double dt = t_new - t_;

    if (dt > 0.0) {
      std::normal_distribution<double> N(0.0, 1.0);
      const double dW = std::sqrt(dt) * N(rng_);

      // OU
      mark_ += cfg_.ou_kappa * (cfg_.ou_theta - mark_) * dt + cfg_.ou_sigma * dW;

      // impact decay
      impact_ *= std::exp(-cfg_.impact_kappa * dt);
      if (cfg_.impact_sigma > 0.0) impact_ += cfg_.impact_sigma * dW;
    }

    t_ = t_new;
  }

  void update_hawkes_state() {
    auto bb = ob_.best_bid();
    auto aa = ob_.best_ask();

    double spread_ticks = 3.0;
    if (bb && aa) spread_ticks = static_cast<double>(*aa - *bb);

    const int db = ob_.depth_size(Side::Bid, 5);
    const int da = ob_.depth_size(Side::Ask, 5);
    const double denom = static_cast<double>(std::max(1, db + da));
    const double imbalance = (static_cast<double>(db) - static_cast<double>(da)) / denom;

    const double edge_ticks =
      static_cast<double>(cfg_.tick.to_ticks(mark_) - cfg_.tick.to_ticks(ref_price()));

    hawkes_.set_state(spread_ticks, imbalance, edge_ticks);
  }

  int sample_level() {
    std::geometric_distribution<int> geo(cfg_.level_geo_p);
    const int k = 1 + geo(rng_);
    return std::min(k, cfg_.max_levels);
  }

  void do_external_event(EType et) {
    std::uniform_real_distribution<double> U(0.0, 1.0);

    const double edge_ticks =
      static_cast<double>(cfg_.tick.to_ticks(mark_) - cfg_.tick.to_ticks(ref_price()));

    auto place_limit = [&](Side side) {
      const int lvl = sample_level();
      const price_t mid = ref_ticks();
      const price_t px = (side == Side::Bid) ? (mid - lvl) : (mid + lvl);
      add_external_limit(side, px, cfg_.ext_limit_qty);
    };

    auto do_cancel = [&](Side side) {
      if (ext_ids_.empty()) return;

      const auto& locs = ob_.locations();

      for (int tries = 0; tries < cfg_.cancel_sample_tries && !ext_ids_.empty(); ++tries) {
        std::uniform_int_distribution<std::size_t> pick(0, ext_ids_.size() - 1);
        const std::uint64_t id = ext_ids_[pick(rng_)];

        auto it = locs.find(id);
        if (it == locs.end()) {
          unregister_external_id(id);
          continue;
        }
        if (it->second.side != side) continue;

        const Order& o = *(it->second.it);
        if ((t_ - o.t_created) < cfg_.min_ext_order_age_s) continue;

        const bool ok = ob_.cancel(id);
        if (ok) unregister_external_id(id);
        return;
      }
    };

    auto do_market = [&](Side side) {
      bool do_it = true;

      // informed filter
      if (U(rng_) < cfg_.p_informed) {
        if (side == Side::Bid) do_it = (edge_ticks > cfg_.info_threshold_ticks);
        else do_it = (edge_ticks < -cfg_.info_threshold_ticks);
      }
      if (!do_it) return;

      const std::uint64_t taker_id = new_id();
      ob_.add_market(taker_id, Owner::External, side, cfg_.lot_size, t_);

      // impact in direction of taker
      impact_ += (side == Side::Bid ? +1.0 : -1.0) * cfg_.impact_per_mkt;
    };

    switch (et) {
      case EType::LimAddBid: place_limit(Side::Bid); break;
      case EType::LimAddAsk: place_limit(Side::Ask); break;
      case EType::CancelBid: do_cancel(Side::Bid); break;
      case EType::CancelAsk: do_cancel(Side::Ask); break;
      case EType::MktBuy:    do_market(Side::Bid); break;
      case EType::MktSell:   do_market(Side::Ask); break;
      default: break;
    }

    // Event-driven requote check
    push_action({t_, next_seq_++, ActionType::RequoteCheck});
  }

  void do_action(const Action& a) {
    if (a.type == ActionType::Cancel) {
      const bool ok = ob_.cancel(a.order_id);
      mm_.note_quote_canceled(a.side, a.order_id, ok);
      return;
    }

    if (a.type == ActionType::Place) {
      Order o{};
      o.id = a.order_id;
      o.owner = a.owner;
      o.side = a.side;
      o.price_ticks = a.price_ticks;
      o.qty = a.qty;
      o.t_created = t_;

      const int remaining = ob_.add_limit(o, /*taker_order_id_for_cross=*/o.id);
      if (remaining > 0) {
        mm_.note_quote_placed(o.side, o.id, o.price_ticks, t_);
      }
      return;
    }

    if (a.type == ActionType::RequoteCheck) {
      maybe_requote();
      return;
    }
  }

  void maybe_requote() {
    auto bb = ob_.best_bid();
    auto aa = ob_.best_ask();

    const price_t mark_mid = mark_ticks();
    const auto plan = mm_.plan_quotes(t_, mark_mid, bb, aa);

    if (plan.update_bid) {
      // Cancel old if present
      if (mm_.bid_id() != 0) {
        Action c{};
        c.t = t_ + cfg_.mm.cancel_latency_s;
        c.seq = next_seq_++;
        c.type = ActionType::Cancel;
        c.order_id = mm_.bid_id();
        c.side = Side::Bid;
        push_action(c);
      }

      Action p{};
      p.t = t_ + cfg_.mm.place_latency_s;
      if (mm_.bid_id() != 0) p.t = t_ + cfg_.mm.cancel_latency_s + cfg_.mm.place_latency_s;
      p.seq = next_seq_++;
      p.type = ActionType::Place;
      p.order_id = new_id();
      p.side = Side::Bid;
      p.price_ticks = plan.new_bid_ticks;
      p.qty = cfg_.mm.quote_qty;
      p.owner = Owner::MM;
      push_action(p);
    }

    if (plan.update_ask) {
      if (mm_.ask_id() != 0) {
        Action c{};
        c.t = t_ + cfg_.mm.cancel_latency_s;
        c.seq = next_seq_++;
        c.type = ActionType::Cancel;
        c.order_id = mm_.ask_id();
        c.side = Side::Ask;
        push_action(c);
      }

      Action p{};
      p.t = t_ + cfg_.mm.place_latency_s;
      if (mm_.ask_id() != 0) p.t = t_ + cfg_.mm.cancel_latency_s + cfg_.mm.place_latency_s;
      p.seq = next_seq_++;
      p.type = ActionType::Place;
      p.order_id = new_id();
      p.side = Side::Ask;
      p.price_ticks = plan.new_ask_ticks;
      p.qty = cfg_.mm.quote_qty;
      p.owner = Owner::MM;
      push_action(p);
    }
  }

  void record(std::int64_t step) {
    TelemetryRow r{};
    r.step = step;
    r.t = t_;

    const price_t mk = mark_ticks();
    r.mid_mark = cfg_.tick.to_price(mk);

    auto bb = ob_.best_bid();
    auto aa = ob_.best_ask();

    if (bb) { r.bid = cfg_.tick.to_price(*bb); r.bid_missing = 0; } else { r.bid_missing = 1; }
    if (aa) { r.ask = cfg_.tick.to_price(*aa); r.ask_missing = 0; } else { r.ask_missing = 1; }

    if (bb && aa) {
      r.spread = cfg_.tick.to_price(*aa - *bb);
      const int bs = ob_.best_bid_size();
      const int as = ob_.best_ask_size();
      r.bid_sz = bs;
      r.ask_sz = as;
      if (bs + as > 0) {
        r.microprice = (r.ask * static_cast<double>(bs) + r.bid * static_cast<double>(as))
                     / static_cast<double>(bs + as);
      }
      r.imbalance = (static_cast<double>(bs) - static_cast<double>(as))
                  / static_cast<double>(std::max(1, bs + as));
    }

    r.inventory = mm_.inventory();
    r.cash = mm_.cash();
    r.inv_value = static_cast<double>(r.inventory) * r.mid_mark;
    r.realized_pnl = mm_.realized_pnl();
    r.unreal_pnl = mm_.unreal_pnl(mk);
    r.fees_paid = mm_.fees_paid();

    r.equity = mm_.equity(mk);
    r.mm_fills = mm_.fills();
    r.mm_cancels_sub = mm_.cancels_submitted();
    r.mm_cancels_ok = mm_.cancels_succeeded();

    if (mm_.bid_price()) r.mm_bid_px = cfg_.tick.to_price(*mm_.bid_price());
    if (mm_.ask_price()) r.mm_ask_px = cfg_.tick.to_price(*mm_.ask_price());

    // TOB share diagnostics
    const auto& locs = ob_.locations();
    if (mm_.bid_id() != 0) {
      auto it = locs.find(mm_.bid_id());
      if (it != locs.end()) r.mm_bid_qty = it->second.it->qty;
    }
    if (mm_.ask_id() != 0) {
      auto it = locs.find(mm_.ask_id());
      if (it != locs.end()) r.mm_ask_qty = it->second.it->qty;
    }

    if (bb && mm_.bid_price() && *mm_.bid_price() == *bb) {
      r.mm_at_best_bid = 1;
      if (r.bid_sz > 0) r.tob_share_bid = static_cast<double>(r.mm_bid_qty) / static_cast<double>(r.bid_sz);
    }
    if (aa && mm_.ask_price() && *mm_.ask_price() == *aa) {
      r.mm_at_best_ask = 1;
      if (r.ask_sz > 0) r.tob_share_ask = static_cast<double>(r.mm_ask_qty) / static_cast<double>(r.ask_sz);
    }

    r.last_trade_px = last_trade_px_;
    r.last_trade_sign = last_trade_sign_;

    telemetry_.push_back(r);
  }

  void dump_csv(const std::string& path) const {
    std::FILE* fp = std::fopen(path.c_str(), "w");
    if (!fp) {
      std::perror("fopen");
      return;
    }

    // Legacy columns first so existing Streamlit logic keeps working.
    std::fprintf(fp,
      "step,equity,inventory,bid,ask,bid_missing,ask_missing,"
      "t,mid_mark,microprice,spread,bid_sz,ask_sz,imbalance,"
      "cash,inv_value,realized_pnl,unreal_pnl,fees_paid,"
      "mm_fills,mm_cancels_sub,mm_cancels_ok,mm_bid_px,mm_ask_px,"
      "mm_bid_qty,mm_ask_qty,mm_at_best_bid,mm_at_best_ask,tob_share_bid,tob_share_ask,"
      "last_trade_px,last_trade_sign\n");

    auto p = [](double x) -> double { return x; };

    for (const auto& r : telemetry_) {
      std::fprintf(fp,
        "%lld,%.10f,%d,%.10f,%.10f,%d,%d,"
        "%.6f,%.10f,%.10f,%.10f,%d,%d,%.6f,"
        "%.10f,%.10f,%.10f,%.10f,%.10f,"
        "%llu,%llu,%llu,%.10f,%.10f,"
        "%d,%d,%d,%d,%.10f,%.10f,"
        "%.10f,%d\n",
        static_cast<long long>(r.step),
        p(r.equity), r.inventory, p(r.bid), p(r.ask), r.bid_missing, r.ask_missing,
        p(r.t), p(r.mid_mark), p(r.microprice), p(r.spread), r.bid_sz, r.ask_sz, p(r.imbalance),
        p(r.cash), p(r.inv_value), p(r.realized_pnl), p(r.unreal_pnl), p(r.fees_paid),
        static_cast<unsigned long long>(r.mm_fills),
        static_cast<unsigned long long>(r.mm_cancels_sub),
        static_cast<unsigned long long>(r.mm_cancels_ok),
        p(r.mm_bid_px), p(r.mm_ask_px),
        r.mm_bid_qty, r.mm_ask_qty, r.mm_at_best_bid, r.mm_at_best_ask, p(r.tob_share_bid), p(r.tob_share_ask),
        p(r.last_trade_px), r.last_trade_sign
      );
    }

    std::fclose(fp);
  }

  void dump_trades_csv(const std::string& path) const {
    std::FILE* fp = std::fopen(path.c_str(), "w");
    if (!fp) {
      std::perror("fopen");
      return;
    }

    std::fprintf(fp, "t,px,qty,sign,mm_involved,mid_mark,spread,maker_age_s,maker_queue_ahead_qty_entry\n");
    for (const auto& tr : trades_) {
      std::fprintf(fp, "%.6f,%.10f,%d,%d,%d,%.10f,%.10f,%.6f,%d\n",
        tr.t, tr.px, tr.qty, tr.sign, tr.mm_involved, tr.mid_mark, tr.spread, tr.maker_age_s, tr.maker_queue_ahead_qty_entry);
    }

    std::fclose(fp);
  }
};

} // namespace sim

int main() {
  sim::SimConfig cfg{};

  // Keep old defaults close to your v4_5, but event-time now uses log_dt.
  cfg.steps = 3000;
  cfg.log_dt = 0.10;
  cfg.tick = sim::TickSpec{0.01};
  cfg.start_price = 100.0;
  cfg.seed = 42;

  // MM defaults tuned to not spam cancels, and to show inventory skew.
  cfg.mm.quote_qty = 1;
  cfg.mm.base_spread_ticks = 2.0;
  cfg.mm.inv_skew_ticks_per_share = 0.18;
  cfg.mm.spread_widen_ticks_per_share = 0.06;
  cfg.mm.inv_requote_band = 5;
  cfg.mm.max_quote_age_s = 0.75;
  cfg.mm.min_quote_life_s = 0.05;
  cfg.mm.cancel_latency_s = 0.03;
  cfg.mm.place_latency_s = 0.01;
  cfg.mm.maker_rebate_per_share = -0.0002;

  // External ecology
  cfg.max_levels = 14;
  cfg.level_geo_p = 0.35;
  cfg.ext_limit_qty = 1;
  cfg.p_informed = 0.22;
  cfg.info_threshold_ticks = 1.0;
  cfg.impact_per_mkt = 0.015;

  // Hawkes-ish flow
  cfg.hawkes.beta = 6.0;
  cfg.hawkes.mu_lim = 85.0;
  cfg.hawkes.mu_cancel = 38.0;
  cfg.hawkes.mu_mkt = 12.0;
  cfg.hawkes.alpha_self = 0.7;
  cfg.hawkes.alpha_cross = 0.15;
  cfg.hawkes.lambda_cap = 900.0;

  sim::Simulator sim(cfg);
  sim.run();
  return 0;
}
