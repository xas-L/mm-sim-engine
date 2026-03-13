"""
Streamlit diagnostic viewer for market-sim v3.0.

Loads:
  telemetry_v3_hawkes.csv   or   telemetry_v3_replay.csv   (per-step snapshots)
  trades_v3_hawkes.csv      or   trades_v3_replay.csv       (per-fill tape)
  multi_seed_summary_v3.csv                                 (optional)

New vs v2.2 viewer:
  Pricing Signals tab: microprice tracking vs BBO mid, and the imbalance EMA
    that drives the MM's fair value — on separate axes, in the correct units.
  Latency distribution panel: log-scale x-axis with fitted log-normal overlay.
    Mean vs 95th percentile gap is the whole point; it needs to be visible.
  Hawkes intensity panel: intensity time-series paired with latency tail plot
    to show the congestion mechanism in action when one is enabled.
  Mode detection: handles both hawkes and replay telemetry transparently.
  Latency metrics in top-bar: mean cancel latency and 95th percentile.
"""

from __future__ import annotations

import io
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats as scipy_stats
import streamlit as st

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.dpi": 130,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

NaN = float("nan")

COL_BID = "#2196F3"
COL_ASK = "#E53935"
COL_MID = "#37474F"
COL_POS = "#43A047"
COL_NEG = "#E53935"
COL_NEU = "#90A4AE"
COL_FV  = "#8E24AA"   # fair value / imbalance signal
COL_HWK = "#E65100"   # Hawkes intensity


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_telemetry(raw: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(raw))
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if "step" not in df.columns:
        df["step"] = np.arange(len(df), dtype=float)
    df = df.sort_values("step").reset_index(drop=True)

    bid_arr = df["bid"].to_numpy(float) if "bid" in df.columns else np.full(len(df), NaN)
    ask_arr = df["ask"].to_numpy(float) if "ask" in df.columns else np.full(len(df), NaN)
    bbo_valid = np.isfinite(bid_arr) & np.isfinite(ask_arr)
    mid_bbo = np.where(bbo_valid, 0.5 * (bid_arr + ask_arr), NaN)

    if "mid_mark" in df.columns:
        df["mid"] = np.where(np.isfinite(mid_bbo), mid_bbo, df["mid_mark"].to_numpy(float))
    else:
        df["mid"] = mid_bbo

    df["bbo_valid"] = bbo_valid.astype(np.int8)

    if "spread" not in df.columns:
        df["spread"] = np.where(bbo_valid, ask_arr - bid_arr, NaN)

    if "equity" in df.columns:
        eq = df["equity"].to_numpy(float)
        step_pnl = np.empty_like(eq)
        step_pnl[0] = NaN
        step_pnl[1:] = eq[1:] - eq[:-1]
        df["step_pnl"] = step_pnl
        safe_eq = np.where(np.isfinite(eq), eq, -np.inf)
        peak = np.maximum.accumulate(safe_eq)
        df["equity_peak"] = np.where(np.isfinite(eq), peak, NaN)
        df["drawdown"] = df["equity_peak"] - df["equity"]

    return df


@st.cache_data(show_spinner=False)
def load_trades(raw: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(raw))
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if "t" in df.columns:
        df = df.sort_values("t").reset_index(drop=True)
    return df


@st.cache_data(show_spinner=False)
def load_seed_summary(raw: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(raw))
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

# Markout

def compute_markout(
    trades: pd.DataFrame,
    tele: pd.DataFrame,
    horizons_s: List[float],
) -> pd.DataFrame:
    """
    Short-horizon markout for MM fills. Convention (maker perspective):
        markout_H = sign * (fill_px - mid(t + H))
    Positive = profitable. Negative = adversely selected.
    """
    if "mm_involved" not in trades.columns:
        return pd.DataFrame()
    mm = trades[trades["mm_involved"] == 1].copy()
    if mm.empty:
        return mm.reset_index(drop=True)
    if "t" not in mm.columns or "t" not in tele.columns or "mid" not in tele.columns:
        return mm.reset_index(drop=True)

    t_ref = tele["t"].to_numpy(float)
    m_ref = tele["mid"].to_numpy(float)
    keep  = np.isfinite(t_ref) & np.isfinite(m_ref)
    t_ref, m_ref = t_ref[keep], m_ref[keep]
    if len(t_ref) < 2:
        return mm.reset_index(drop=True)

    t_fill  = mm["t"].to_numpy(float)
    px_fill = mm["px"].to_numpy(float)
    sign    = mm["sign"].to_numpy(float)

    for H in horizons_s:
        mid_future = np.interp(t_fill + H, t_ref, m_ref, left=NaN, right=NaN)
        mm[f"markout_{H:.0f}s"] = sign * (px_fill - mid_future)

    return mm.reset_index(drop=True)


def enrich_fills_with_tele_state(
    mm_fills: pd.DataFrame,
    tele: pd.DataFrame,
) -> pd.DataFrame:
    if mm_fills.empty or tele.empty:
        return mm_fills
    if "t" not in mm_fills.columns or "t" not in tele.columns:
        return mm_fills
    cols_to_add = [c for c in ["imbalance", "spread", "microprice", "mid"]
                   if c in tele.columns]
    if not cols_to_add:
        return mm_fills
    tele_sub = tele[["t"] + cols_to_add].copy().sort_values("t").rename(
        columns={c: f"tele_{c}" for c in cols_to_add}
    )
    return pd.merge_asof(
        mm_fills.sort_values("t"), tele_sub, on="t", direction="backward",
    ).sort_values("t").reset_index(drop=True)

# Rolling IR

def sim_time_sharpe(step_pnl: pd.Series, window_s: float, log_dt: float) -> pd.Series:
    x = step_pnl.astype(float)
    w = max(5, int(round(window_s / log_dt)))
    mu = x.rolling(w, min_periods=max(3, w // 4)).mean()
    sd = x.rolling(w, min_periods=max(3, w // 4)).std(ddof=1)
    ir = (mu / sd) * np.sqrt(w)
    ir[sd < 1e-12] = NaN
    return ir

# Helpers

def has_col(df: pd.DataFrame, *cols: str) -> bool:
    return all(c in df.columns for c in cols)


def safe_bar_colours(values) -> List[str]:
    return [COL_POS if (np.isfinite(v) and v > 0) else COL_NEG for v in values]


def grouped_mean_bar(
    ax: plt.Axes,
    series: pd.Series,
    target: pd.Series,
    bins,
    labels: List[str],
    xlabel: str,
    ylabel: str,
    title: str,
) -> None:
    groups = pd.cut(series, bins=bins, labels=labels, right=False)
    means  = target.groupby(groups, observed=True).mean()
    counts = target.groupby(groups, observed=True).count()
    valid  = counts >= 2
    means  = means[valid]
    if means.empty:
        ax.set_visible(False)
        return
    ax.bar(means.index.astype(str), means.values,
           color=safe_bar_colours(means.values), alpha=0.82,
           edgecolor="white", linewidth=0.5)
    ax.axhline(0, color=COL_MID, lw=0.8, ls="--")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis="x", labelsize=8)

# Figure functions — carried forward from v2.2

def fig_equity_drawdown(df: pd.DataFrame, xcol: str, xlab: str) -> plt.Figure:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    x = df[xcol]
    ax1.plot(x, df["equity"], lw=1.3, color=COL_BID, label="Equity")
    if has_col(df, "equity_peak"):
        ax1.plot(x, df["equity_peak"], lw=0.8, color=COL_NEU, ls="--", alpha=0.7, label="Peak")
    ax1.axhline(0, color=COL_MID, lw=0.7, ls="--")
    ax1.set_ylabel("Equity ($)")
    ax1.set_title("Equity Curve")
    ax1.legend()
    if has_col(df, "drawdown"):
        ax2.fill_between(x, df["drawdown"], color=COL_NEG, alpha=0.45, lw=0)
        ax2.plot(x, df["drawdown"], lw=0.7, color=COL_NEG, alpha=0.7)
        ax2.set_ylabel("Drawdown ($)")
        ax2.set_xlabel(xlab)
        ax2.set_title("Drawdown from Peak")
    fig.tight_layout()
    return fig


def fig_inventory(df: pd.DataFrame, xcol: str, xlab: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 3.5))
    x = df[xcol]
    inv = df["inventory"].to_numpy(float)
    ax.step(x, inv, where="post", color="#F57C00", lw=1.0)
    ax.fill_between(x, inv, step="post", alpha=0.25, color="#F57C00")
    ax.axhline(0, color=COL_MID, lw=0.7, ls="--")
    ax.set_ylabel("Inventory (shares)")
    ax.set_xlabel(xlab)
    ax.set_title("Inventory Over Time")
    fig.tight_layout()
    return fig


def fig_pnl_components(df: pd.DataFrame, xcol: str, xlab: str) -> Optional[plt.Figure]:
    if not has_col(df, "realized_pnl", "unreal_pnl", "fees_paid"):
        return None
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    x = df[xcol]
    axes[0].plot(x, df["realized_pnl"], color=COL_POS, lw=1.1)
    axes[0].axhline(0, color=COL_MID, lw=0.6, ls="--")
    axes[0].set_ylabel("Realised P&L ($)")
    axes[0].set_title("P&L Components")
    axes[1].plot(x, df["unreal_pnl"], color=COL_BID, lw=1.0)
    axes[1].axhline(0, color=COL_MID, lw=0.6, ls="--")
    axes[1].set_ylabel("Unrealised P&L ($)")
    axes[2].plot(x, df["fees_paid"], color="#7B1FA2", lw=1.0)
    axes[2].set_ylabel("Cumulative Fees ($)")
    axes[2].set_xlabel(xlab)
    fig.tight_layout()
    return fig


def fig_rolling_ir(
    df: pd.DataFrame, xcol: str, xlab: str, window_s: float, log_dt: float,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 3.5))
    if has_col(df, "step_pnl"):
        ir = sim_time_sharpe(df["step_pnl"], window_s, log_dt)
        ax.plot(df[xcol], ir, lw=1.0, color=COL_BID)
        ax.axhline(0,  color=COL_MID, lw=0.7, ls="--")
        ax.axhline(1,  color=COL_POS, lw=0.6, ls="--", alpha=0.6, label="IR = 1")
        ax.axhline(-1, color=COL_NEG, lw=0.6, ls="--", alpha=0.6)
        ax.legend(fontsize=8)
    w_steps = int(round(window_s / log_dt))
    ax.set_ylabel(f"IR ({window_s:.0f}s window)")
    ax.set_xlabel(xlab)
    ax.set_title(
        f"Rolling Information Ratio  |  window = {window_s:.0f}s simulated "
        f"({w_steps} steps)  |  NOT annualised to calendar time"
    )
    fig.tight_layout()
    return fig


def fig_tob_quotes(
    df: pd.DataFrame, xcol: str, xlab: str, sample_every: int = 1
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 4))
    ds = df.iloc[::max(1, sample_every)]
    x  = ds[xcol]
    if "bid" in ds.columns:
        ax.plot(x, ds["bid"], lw=0.7, color=COL_BID, alpha=0.7, label="Best bid")
    if "ask" in ds.columns:
        ax.plot(x, ds["ask"], lw=0.7, color=COL_ASK, alpha=0.7, label="Best ask")
    if "mid_mark" in ds.columns:
        ax.plot(x, ds["mid_mark"], lw=0.9, color=COL_MID, alpha=0.7, ls="--",
                label="OU fundamental (unobservable by MM)")
    if "mm_bid_px" in ds.columns:
        valid = ds["mm_bid_px"].notna()
        ax.scatter(x[valid], ds.loc[valid, "mm_bid_px"], s=4,
                   color=COL_BID, alpha=0.45, marker="v", label="MM bid")
    if "mm_ask_px" in ds.columns:
        valid = ds["mm_ask_px"].notna()
        ax.scatter(x[valid], ds.loc[valid, "mm_ask_px"], s=4,
                   color=COL_ASK, alpha=0.45, marker="^", label="MM ask")
    ax.set_xlabel(xlab)
    ax.set_ylabel("Price ($)")
    ax.set_title("Top of Book: BBO and MM Quotes")
    ax.legend(ncol=3)
    fig.tight_layout()
    return fig


def fig_markout_profile(mm_fills: pd.DataFrame, horizons: List[float]) -> Optional[plt.Figure]:
    h_cols = [f"markout_{H:.0f}s" for H in horizons if f"markout_{H:.0f}s" in mm_fills.columns]
    if not h_cols:
        return None
    labels = [c.replace("markout_", "") for c in h_cols]
    means  = [mm_fills[c].mean() for c in h_cols]
    stds   = [mm_fills[c].std()  for c in h_cols]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.bar(labels, means, color=safe_bar_colours(means), alpha=0.82, edgecolor="white")
    ax1.errorbar(labels, means, yerr=stds, fmt="none", color=COL_MID, capsize=5, lw=1.2)
    ax1.axhline(0, color=COL_MID, lw=0.8, ls="--")
    ax1.set_xlabel("Horizon (simulated seconds)")
    ax1.set_ylabel("Mean markout ($)")
    ax1.set_title(
        f"MM Fill Markout Profile  (n = {len(mm_fills):,})\n"
        "Positive = profitable, negative = adverse selection"
    )
    c0   = h_cols[0]
    vals = mm_fills[c0].dropna()
    ax2.hist(vals, bins=50, color=COL_BID, alpha=0.72, edgecolor="white")
    ax2.axvline(0, color=COL_MID, lw=1.1, ls="--")
    mu_val = vals.mean()
    ax2.axvline(mu_val, color=COL_NEG if mu_val < 0 else COL_POS,
                lw=1.5, ls="--", label=f"Mean = {mu_val:.5f}")
    ax2.set_xlabel(f"Markout at {labels[0]} ($)")
    ax2.set_ylabel("Count")
    ax2.set_title(f"Distribution at {labels[0]}")
    ax2.legend()
    fig.tight_layout()
    return fig


def fig_cumulative_markout(mm_fills: pd.DataFrame, horizons: List[float]) -> Optional[plt.Figure]:
    h_cols = [f"markout_{H:.0f}s" for H in horizons if f"markout_{H:.0f}s" in mm_fills.columns]
    if not h_cols or "t" not in mm_fills.columns:
        return None
    fig, ax = plt.subplots(figsize=(12, 4))
    sorted_mm   = mm_fills.sort_values("t")
    colours_line = plt.cm.Blues(np.linspace(0.45, 0.9, len(h_cols)))
    for col, colour in zip(h_cols, colours_line):
        label = col.replace("markout_", "").replace("s", "s horizon")
        ax.plot(sorted_mm["t"], sorted_mm[col].fillna(0).cumsum(),
                lw=1.2, color=colour, label=label)
    ax.axhline(0, color=COL_MID, lw=0.7, ls="--")
    ax.set_xlabel("Simulated time (s)")
    ax.set_ylabel("Cumulative markout ($)")
    ax.set_title("Cumulative Markout of MM Fills by Horizon")
    ax.legend()
    fig.tight_layout()
    return fig


def fig_fill_direction(mm_fills: pd.DataFrame) -> Optional[plt.Figure]:
    if not has_col(mm_fills, "sign"):
        return None
    h_col = next(
        (c for c in [f"markout_{H:.0f}s" for H in [5, 1, 10, 30]]
         if c in mm_fills.columns and mm_fills[c].notna().sum() >= 5),
        None,
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    n_buys  = int((mm_fills["sign"] == -1).sum())
    n_sells = int((mm_fills["sign"] == +1).sum())
    axes[0].bar(["MM bought\n(taker sold)", "MM sold\n(taker bought)"],
                [n_buys, n_sells], color=[COL_BID, COL_ASK], alpha=0.82, edgecolor="white")
    axes[0].set_ylabel("Fill count")
    axes[0].set_title("Fill Direction Breakdown")
    if h_col:
        by_sign = mm_fills.groupby("sign")[h_col]
        for sign_val, colour, label in [(-1, COL_BID, "MM bought (sign=-1)"),
                                         (+1, COL_ASK, "MM sold (sign=+1)")]:
            grp = by_sign.get_group(sign_val).dropna() if sign_val in by_sign.groups else pd.Series(dtype=float)
            if len(grp) > 2:
                axes[1].hist(grp, bins=40, alpha=0.6, color=colour, edgecolor="white", label=label)
        axes[1].axvline(0, color=COL_MID, lw=1.1, ls="--")
        axes[1].set_xlabel(f"{h_col} ($)")
        axes[1].set_ylabel("Count")
        axes[1].set_title(f"Markout Distribution by Fill Direction ({h_col})")
        axes[1].legend()
    else:
        axes[1].set_visible(False)
    fig.tight_layout()
    return fig


def fig_adverse_selection(mm_fills: pd.DataFrame) -> Optional[plt.Figure]:
    h_col = next(
        (c for c in [f"markout_{H:.0f}s" for H in [5, 1, 10, 30]]
         if c in mm_fills.columns and mm_fills[c].notna().sum() >= 10),
        None,
    )
    if h_col is None:
        return None
    y   = mm_fills[h_col]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()
    fig.suptitle(
        f"Adverse Selection Stratification  |  target = {h_col}\n"
        "Green = profitable on average at this horizon, red = adversely selected",
        fontsize=11,
    )
    if has_col(mm_fills, "maker_queue_ahead_qty_entry"):
        grouped_mean_bar(axes[0], mm_fills["maker_queue_ahead_qty_entry"].fillna(0).clip(0), y,
                         bins=[0, 1, 3, 7, 15, 9999], labels=["0", "1-2", "3-6", "7-14", "15+"],
                         xlabel="Queue ahead at entry (shares)", ylabel=f"Mean {h_col} ($)",
                         title="Adverse Selection vs Queue Position")
    if has_col(mm_fills, "maker_age_s"):
        grouped_mean_bar(axes[1], mm_fills["maker_age_s"].fillna(0).clip(0), y,
                         bins=[0, 0.05, 0.15, 0.35, 0.75, 9999],
                         labels=["<0.05s", "0.05-0.15s", "0.15-0.35s", "0.35-0.75s", ">0.75s"],
                         xlabel="Order age at fill (simulated seconds)", ylabel=f"Mean {h_col} ($)",
                         title="Adverse Selection vs Maker Age")
    if has_col(mm_fills, "queue_depleted_before_fill"):
        grouped_mean_bar(axes[2], mm_fills["queue_depleted_before_fill"].fillna(0).clip(0), y,
                         bins=[0, 1, 3, 7, 15, 9999], labels=["0", "1-2", "3-6", "7-14", "15+"],
                         xlabel="Volume cleared ahead before fill", ylabel=f"Mean {h_col} ($)",
                         title="Adverse Selection vs Queue Depletion")
    imb_col = "tele_imbalance" if "tele_imbalance" in mm_fills.columns else "imbalance"
    if imb_col in mm_fills.columns:
        grouped_mean_bar(axes[3], mm_fills[imb_col].fillna(0), y,
                         bins=[-1.01, -0.5, -0.2, 0.2, 0.5, 1.01],
                         labels=["<-0.5", "-0.5/-.2", "-.2/.2", ".2/.5", ">.5"],
                         xlabel="Book imbalance at fill", ylabel=f"Mean {h_col} ($)",
                         title="Adverse Selection vs Imbalance at Fill")
    else:
        axes[3].set_visible(False)
    fig.tight_layout()
    return fig


def fig_toxicity(df: pd.DataFrame, xcol: str, xlab: str) -> Optional[plt.Figure]:
    have = [c for c in ["fill_sign_ema", "tox_widen"] if c in df.columns]
    if not have:
        return None
    fig, axes = plt.subplots(len(have), 1, figsize=(12, 3.5 * len(have)), sharex=True)
    if len(have) == 1:
        axes = [axes]
    x = df[xcol]
    if "fill_sign_ema" in df.columns:
        ax  = axes[0]
        ema = df["fill_sign_ema"]
        ax.plot(x, ema, lw=0.9, color="#F57C00")
        ax.fill_between(x, ema, color="#F57C00", alpha=0.15)
        ax.axhline(0,    color=COL_MID, lw=0.7, ls="--")
        ax.axhline(0.4,  color=COL_NEG, lw=0.7, ls="--", alpha=0.6, label="Threshold +0.4")
        ax.axhline(-0.4, color=COL_NEG, lw=0.7, ls="--", alpha=0.6, label="Threshold -0.4")
        ax.set_ylabel("Fill direction EMA")
        ax.set_title("Toxicity Signal (EMA of fill direction)\n+1 = all buys, -1 = all sells")
        ax.set_ylim(-1.1, 1.1)
        ax.legend()
    if "tox_widen" in df.columns and len(axes) > 1:
        ax = axes[1]
        ax.fill_between(x, df["tox_widen"], color=COL_NEG, alpha=0.45, lw=0)
        ax.plot(x, df["tox_widen"], lw=0.8, color=COL_NEG, alpha=0.8)
        ax.set_ylabel("Tox widen (ticks)")
        ax.set_xlabel(xlab)
        ax.set_title("Adaptive Spread Widening from Toxicity Signal")
    fig.tight_layout()
    return fig


def fig_tox_vs_markout(mm_fills: pd.DataFrame) -> Optional[plt.Figure]:
    if not has_col(mm_fills, "mm_tox_widen"):
        return None
    h_col = next(
        (c for c in [f"markout_{H:.0f}s" for H in [5, 1, 10, 30]]
         if c in mm_fills.columns and mm_fills[c].notna().sum() >= 10),
        None,
    )
    if h_col is None:
        return None
    fig, ax = plt.subplots(figsize=(8, 5))
    tw = mm_fills["mm_tox_widen"].clip(0, 3)
    mk = mm_fills[h_col].clip(-0.15, 0.15)
    ax.scatter(tw, mk, alpha=0.25, s=12, color=COL_BID, edgecolors="none")
    ax.axhline(0, color=COL_MID, lw=0.8, ls="--")
    bins_tw  = np.linspace(0, 3, 7)
    bin_mids = 0.5 * (bins_tw[:-1] + bins_tw[1:])
    means_tw = mk.groupby(pd.cut(tw, bins=bins_tw), observed=True).mean()
    ax.plot(bin_mids[:len(means_tw)], means_tw.values, "o-",
            color=COL_ASK, lw=1.5, ms=5, label="Bin mean")
    ax.set_xlabel("Toxicity widen active at fill (ticks)")
    ax.set_ylabel(f"{h_col} ($)")
    ax.set_title(
        "Does adaptive widening protect against adverse selection?\n"
        "Downward trend expected if widening is reactive rather than predictive"
    )
    ax.legend()
    fig.tight_layout()
    return fig


def fig_lob_depth(
    df: pd.DataFrame, xcol: str, xlab: str, sample_every: int = 1
) -> Optional[plt.Figure]:
    present_bid = [c for c in [f"bid_l{i}" for i in range(1, 6)] if c in df.columns]
    present_ask = [c for c in [f"ask_l{i}" for i in range(1, 6)] if c in df.columns]
    if not present_bid and not present_ask:
        return None
    df_s = df.iloc[::max(1, sample_every)]
    x    = df_s[xcol]
    fig, (ax_bid, ax_ask) = plt.subplots(2, 1, figsize=(13, 7), sharex=True)

    def stacked(ax, cols, cmap_name, side_label):
        data    = np.nan_to_num(df_s[cols].to_numpy(float), nan=0.0)
        colours = plt.get_cmap(cmap_name)(np.linspace(0.35, 0.85, data.shape[1]))
        bottom  = np.zeros(len(df_s))
        for i in range(data.shape[1]):
            ax.fill_between(x, bottom, bottom + data[:, i],
                            color=colours[i], alpha=0.88, step="post", label=f"L{i+1}")
            bottom += data[:, i]
        ax.set_ylabel(f"{side_label} depth (shares)")
        ax.set_title(f"LOB Depth: {side_label} Side (L1 = best price)")
        ax.legend(fontsize=7, ncol=5, loc="upper right")

    if present_bid:
        stacked(ax_bid, present_bid, "Blues", "Bid")
    if present_ask:
        stacked(ax_ask, present_ask, "Reds", "Ask")
    ax_ask.set_xlabel(xlab)
    fig.tight_layout()
    return fig


def fig_imbalance(df: pd.DataFrame, xcol: str, xlab: str) -> Optional[plt.Figure]:
    if "imbalance" not in df.columns:
        return None
    fig, ax = plt.subplots(figsize=(12, 3))
    x   = df[xcol]
    imb = df["imbalance"].to_numpy(float)
    ax.plot(x, imb, lw=0.7, color=COL_BID, alpha=0.8)
    ax.fill_between(x, imb, color=COL_BID, alpha=0.15)
    ax.axhline(0, color=COL_MID, lw=0.8, ls="--")
    ax.set_ylabel("Imbalance")
    ax.set_xlabel(xlab)
    ax.set_title("Top-of-Book Imbalance  (bid_sz - ask_sz) / (bid_sz + ask_sz)")
    fig.tight_layout()
    return fig


def fig_tob_share(df: pd.DataFrame, xcol: str, xlab: str) -> Optional[plt.Figure]:
    if not has_col(df, "tob_share_bid", "tob_share_ask"):
        return None
    fig, ax = plt.subplots(figsize=(12, 3))
    x = df[xcol]
    ax.scatter(x, df["tob_share_bid"], s=3, alpha=0.3, color=COL_BID, label="Bid TOB share")
    ax.scatter(x, df["tob_share_ask"], s=3, alpha=0.3, color=COL_ASK, label="Ask TOB share")
    ax.set_ylabel("MM share of TOB qty")
    ax.set_xlabel(xlab)
    ax.set_title("MM Queue Share at Best Bid/Ask")
    ax.legend()
    fig.tight_layout()
    return fig


def fig_multi_seed(summary: pd.DataFrame) -> List[plt.Figure]:
    figs = []
    seeds_str = (summary["seed"].astype(int).astype(str)
                 if "seed" in summary.columns else summary.index.astype(str))
    if "final_equity" in summary.columns:
        fig, ax = plt.subplots(figsize=(10, 4))
        fe = summary["final_equity"]
        ax.bar(seeds_str, fe, color=safe_bar_colours(fe.values), alpha=0.82, edgecolor="white")
        ax.axhline(0, color=COL_MID, lw=0.8, ls="--")
        mu = fe.mean()
        ax.axhline(mu, color="#1565C0", lw=1.3, ls="--", label=f"Mean = ${mu:.2f}")
        ax.set_xlabel("Seed")
        ax.set_ylabel("Final equity ($)")
        ax.set_title(f"Final Equity Across Seeds  |  {int((fe > 0).sum())}/{len(fe)} positive")
        ax.legend()
        ax.tick_params(axis="x", rotation=25)
        fig.tight_layout()
        figs.append(fig)
    if "max_drawdown" in summary.columns:
        fig, ax = plt.subplots(figsize=(10, 4))
        dd = summary["max_drawdown"]
        ax.bar(seeds_str, dd, color=COL_NEG, alpha=0.72, edgecolor="white")
        ax.axhline(dd.mean(), color="#1565C0", lw=1.3, ls="--", label=f"Mean = ${dd.mean():.2f}")
        ax.set_xlabel("Seed")
        ax.set_ylabel("Max drawdown ($)")
        ax.set_title("Max Drawdown Across Seeds")
        ax.legend()
        ax.tick_params(axis="x", rotation=25)
        fig.tight_layout()
        figs.append(fig)
    if has_col(summary, "final_equity", "max_drawdown"):
        fig, ax = plt.subplots(figsize=(6, 5))
        fe = summary["final_equity"]
        dd = summary["max_drawdown"]
        ax.scatter(dd, fe, color=safe_bar_colours(fe.values), s=80,
                   edgecolors=COL_MID, linewidths=0.5, zorder=3)
        for i, seed in enumerate(seeds_str):
            ax.annotate(str(seed), (dd.iloc[i], fe.iloc[i]),
                        fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax.axhline(0, color=COL_MID, lw=0.7, ls="--")
        ax.set_xlabel("Max drawdown ($)")
        ax.set_ylabel("Final equity ($)")
        ax.set_title("Equity vs Drawdown by Seed")
        fig.tight_layout()
        figs.append(fig)
    return figs


def fig_accounting_check(df: pd.DataFrame, xcol: str, xlab: str) -> Optional[plt.Figure]:
    if not has_col(df, "equity", "cash", "inv_value"):
        return None
    recon    = df["cash"] + df["inv_value"]
    residual = df["equity"] - recon
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    x = df[xcol]
    ax1.plot(x, df["equity"], lw=1.1, color=COL_BID, label="equity")
    ax1.plot(x, recon, lw=0.8, color=COL_NEG, ls="--", alpha=0.75, label="cash + inv_value")
    ax1.set_ylabel("($)")
    ax1.set_title("Accounting Identity: equity == cash + inv_value")
    ax1.legend()
    ax2.plot(x, residual, lw=0.8, color="#7B1FA2")
    ax2.axhline(0, color=COL_MID, lw=0.7, ls="--")
    ax2.set_ylabel("Residual ($)")
    ax2.set_xlabel(xlab)
    ax2.set_title("Residual (should be near machine epsilon throughout)")
    max_resid = float(residual.abs().max())
    ax2.text(0.01, 0.93, f"Max |residual| = {max_resid:.2e}",
             transform=ax2.transAxes, fontsize=8, va="top",
             color=COL_NEG if max_resid > 1e-4 else COL_POS)
    fig.tight_layout()
    return fig


def fig_pnl_hist_qq(df: pd.DataFrame) -> Optional[plt.Figure]:
    if "step_pnl" not in df.columns:
        return None
    sp = df["step_pnl"].dropna()
    if len(sp) < 4:
        return None
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.hist(sp, bins=60, color=COL_BID, alpha=0.72, edgecolor="white")
    ax1.axvline(0, color=COL_MID, lw=0.9, ls="--")
    mu_val = sp.mean()
    ax1.axvline(mu_val, color=COL_NEG if mu_val < 0 else COL_POS,
                lw=1.4, ls="--", label=f"Mean = {mu_val:.5f}")
    ax1.set_xlabel("Step P&L ($)")
    ax1.set_ylabel("Count")
    ax1.set_title("Per-step P&L Distribution")
    ax1.legend()
    scipy_stats.probplot(sp, dist="norm", plot=ax2)
    ax2.set_title("Q-Q Plot vs Normal")
    fig.tight_layout()
    return fig

# Figure functions — v3.0 new panels

def fig_pricing_signals(df: pd.DataFrame, xcol: str, xlab: str) -> Optional[plt.Figure]:
    """
    Two-panel figure showing the MM's observable fair value signals.

    Top: microprice vs BBO mid vs OU fundamental. The point is that microprice
    tracks mid closely (it's derived from observable quotes), while the OU
    fundamental diverges — the MM does not have access to the fundamental.

    Bottom: raw imbalance EMA (fv_imb_ema). This is the momentum term that
    shifts the reservation price; it lives in [-1, +1] and is plotted on its
    own axis. Overlaying it on the price chart would be wrong — different units.
    """
    has_micro = "microprice" in df.columns
    has_imb   = "fv_imb_ema" in df.columns
    if not has_micro and not has_imb:
        return None

    n_panels = 1 + int(has_imb)
    fig, axes = plt.subplots(n_panels, 1, figsize=(12, 4 * n_panels), sharex=True)
    if n_panels == 1:
        axes = [axes]

    x = df[xcol]

    #  top: price panel 
    ax = axes[0]
    if "mid" in df.columns:
        ax.plot(x, df["mid"], lw=0.9, color=COL_MID, alpha=0.6, label="BBO mid")
    if has_micro:
        ax.plot(x, df["microprice"], lw=1.1, color=COL_FV, alpha=0.85, label="Microprice")
    if "mid_mark" in df.columns:
        ax.plot(x, df["mid_mark"], lw=0.7, color=COL_NEU, alpha=0.5, ls=":",
                label="OU fundamental (latent, unobservable)")
    ax.set_ylabel("Price ($)")
    ax.set_title(
        "Microprice vs BBO Mid — the MM's observable fair value anchor\n"
        "Microprice = (ask·bid_qty + bid·ask_qty) / (bid_qty + ask_qty)"
    )
    ax.legend(ncol=3)

    # bottom: imbalance EMA 
    if has_imb:
        ax2 = axes[1]
        imb = df["fv_imb_ema"].to_numpy(float)
        ax2.plot(x, imb, lw=0.9, color=COL_FV, alpha=0.85)
        ax2.fill_between(x, imb, color=COL_FV, alpha=0.12)
        ax2.axhline(0,    color=COL_MID, lw=0.7, ls="--")
        ax2.axhline(0.5,  color=COL_BID, lw=0.5, ls="--", alpha=0.5)
        ax2.axhline(-0.5, color=COL_ASK, lw=0.5, ls="--", alpha=0.5)
        ax2.set_ylim(-1.05, 1.05)
        ax2.set_ylabel("Imbalance EMA")
        ax2.set_xlabel(xlab)
        ax2.set_title(
            "Imbalance EMA — momentum term in reservation price\n"
            "+1 = sustained bid-heavy book (FV shifted up), -1 = sustained ask-heavy"
        )

    fig.tight_layout()
    return fig


def fig_latency_distribution(trades: pd.DataFrame) -> Optional[plt.Figure]:
    """
    Log-scale histogram of realised cancel and place latency with a fitted
    log-normal overlay. The whole point of log-normal latency is the gap between
    the mode (hardware floor) and the 95th percentile (jitter tail). A linear
    x-axis would compress that tail into invisibility — log scale is mandatory.
    """
    cols = [c for c in ["cancel_latency_s", "place_latency_s"] if c in trades.columns]
    if not cols:
        return None

    # Filter to rows where the MM actually submitted, skipping external fills (lat == 0).
    plot_data: Dict[str, np.ndarray] = {}
    for col in ["cancel_latency_s", "place_latency_s"]:
        if col not in trades.columns:
            continue
        vals = trades[col].dropna().to_numpy(float)
        vals = vals[vals > 1e-6]   # drop zero-latency rows (non-MM fills)
        if len(vals) >= 5:
            plot_data[col] = vals * 1000.0   #  milliseconds

    if not plot_data:
        return None

    ncols = len(plot_data)
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 5), squeeze=False)
    axes = axes[0]

    specs = {
        "cancel_latency_s": ("Cancel Latency", COL_NEG),
        "place_latency_s":  ("Place Latency",  COL_BID),
    }

    for ax, (col, vals) in zip(axes, plot_data.items()):
        label, colour = specs[col]

        # Log-space bins: equal width on log axis.
        lo, hi = np.log10(vals.min()), np.log10(vals.max())
        bins   = np.logspace(lo, hi, 55)

        ax.hist(vals, bins=bins, alpha=0.65, color=colour, edgecolor="white")

        # Fit log-normal and overlay density curve.
        log_vals       = np.log(vals)
        mu_fit, s_fit  = log_vals.mean(), log_vals.std(ddof=1)
        x_fit          = np.logspace(lo, hi, 400)
        pdf_fit        = scipy_stats.lognorm.pdf(x_fit, s=s_fit, scale=np.exp(mu_fit))

        # Scale pdf to match histogram counts.
        bin_widths    = np.diff(bins)
        density_scale = len(vals) * bin_widths.mean()
        ax2 = ax.twinx()
        ax2.plot(x_fit, pdf_fit, lw=1.5, color=COL_MID, ls="--", alpha=0.8,
                 label=f"Log-normal fit\nμ={mu_fit:.2f}, σ={s_fit:.2f}")
        ax2.set_ylabel("Density", fontsize=8)
        ax2.tick_params(axis="y", labelsize=7)
        ax2.legend(fontsize=7, loc="upper right")

        p50  = np.percentile(vals, 50)
        p95  = np.percentile(vals, 95)
        p99  = np.percentile(vals, 99)
        ax.axvline(p50, color=COL_POS, lw=1.2, ls="--",
                   label=f"Median  {p50:.1f} ms")
        ax.axvline(p95, color="#F57C00", lw=1.2, ls="--",
                   label=f"95th    {p95:.1f} ms")
        ax.axvline(p99, color=COL_NEG, lw=1.2, ls="--",
                   label=f"99th    {p99:.1f} ms")

        ax.set_xscale("log")
        ax.set_xlabel("Realised latency (ms, log scale)")
        ax.set_ylabel("Order count")
        ax.set_title(
            f"{label}\n"
            f"n = {len(vals):,}  |  mean = {vals.mean():.1f} ms  |  "
            f"95th = {p95:.1f} ms  |  99th = {p99:.1f} ms"
        )
        ax.legend(fontsize=7, loc="upper left")

    fig.suptitle(
        "Realised Log-Normal Latency Distribution\n"
        "Log-scale x-axis required: linear scale collapses the heavy right tail",
        fontsize=11,
    )
    fig.tight_layout()
    return fig


def fig_hawkes_intensity(df: pd.DataFrame, xcol: str, xlab: str) -> Optional[plt.Figure]:
    """
    Hawkes total intensity over time. Zero-valued columns (replay mode) are
    silently suppressed — the column exists in replay telemetry but carries no
    information since the generative process is not running.
    """
    if "hawkes_intensity" not in df.columns:
        return None
    intensity = df["hawkes_intensity"].to_numpy(float)
    if intensity.max() < 1.0:
        # Replay mode: all zeros. Nothing useful to plot.
        return None
    fig, ax = plt.subplots(figsize=(12, 3))
    x = df[xcol]
    ax.plot(x, intensity, lw=0.9, color=COL_HWK)
    ax.fill_between(x, intensity, color=COL_HWK, alpha=0.15)
    ax.set_xlabel(xlab)
    ax.set_ylabel("Intensity (events/sec)")
    ax.set_title(
        "Hawkes Flow Intensity — total λ(t) over time\n"
        "Burst periods (spikes) drive queue congestion delay when congestion_per_event_s > 0"
    )
    fig.tight_layout()
    return fig


def fig_latency_vs_intensity(
    trades: pd.DataFrame, tele: pd.DataFrame
) -> Optional[plt.Figure]:
    """
    Scatter of realised cancel latency vs Hawkes intensity at fill time.
    Only meaningful when congestion model is on; shows that latency spikes
    during high-intensity periods (the mechanism we built).
    If congestion is disabled, the scatter will be a horizontal band.
    """
    if "cancel_latency_s" not in trades.columns or "hawkes_intensity" not in tele.columns:
        return None
    intensity = tele["hawkes_intensity"].to_numpy(float)
    if intensity.max() < 1.0:
        return None

    # Join cancel latency at each fill time to Hawkes intensity at that time.
    lat = trades[["t", "cancel_latency_s"]].dropna().copy()
    lat = lat[lat["cancel_latency_s"] > 1e-6]
    if len(lat) < 20:
        return None

    t_ref  = tele["t"].to_numpy(float)
    i_ref  = tele["hawkes_intensity"].to_numpy(float)
    i_at_t = np.interp(lat["t"].to_numpy(float), t_ref, i_ref)
    lat_ms = lat["cancel_latency_s"].to_numpy(float) * 1000.0

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(i_at_t, lat_ms, alpha=0.2, s=10, color=COL_HWK, edgecolors="none")

    # Bin mean overlay.
    bins   = np.percentile(i_at_t, np.linspace(0, 100, 9))
    bins   = np.unique(bins)
    if len(bins) > 2:
        mids  = 0.5 * (bins[:-1] + bins[1:])
        means = [lat_ms[
            (i_at_t >= bins[k]) & (i_at_t < bins[k + 1])
        ].mean() for k in range(len(bins) - 1)]
        valid = [np.isfinite(m) for m in means]
        ax.plot(np.array(mids)[valid], np.array(means)[valid], "o-",
                color=COL_MID, lw=1.5, ms=5, label="Bin mean")

    ax.set_xlabel("Hawkes intensity at fill time (events/sec)")
    ax.set_ylabel("Realised cancel latency (ms)")
    ax.set_title(
        "Cancel Latency vs Order Flow Intensity\n"
        "Upward trend = congestion model active; flat = congestion disabled"
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig

# Diagnostics

def compute_diagnostics(df: pd.DataFrame) -> Dict:
    d: Dict = {}
    if "step_pnl" in df.columns:
        sp = df["step_pnl"].dropna()
        d["n_steps"]    = len(df)
        d["n_nonzero"]  = int((sp != 0).sum())
        d["mean_pnl"]   = float(sp.mean())   if len(sp) > 0 else NaN
        d["std_pnl"]    = float(sp.std(ddof=1)) if len(sp) > 1 else NaN
        d["median_pnl"] = float(sp.median()) if len(sp) > 0 else NaN
        d["mad_pnl"]    = float(np.median(np.abs(sp - sp.median()))) if len(sp) > 0 else NaN
    if "equity"   in df.columns: d["final_equity"] = float(df["equity"].iloc[-1])
    if "drawdown" in df.columns: d["max_drawdown"]  = float(df["drawdown"].max())
    if "bbo_valid" in df.columns: d["bbo_rate"] = float(df["bbo_valid"].mean())
    if has_col(df, "equity", "cash", "inv_value"):
        d["max_acct_resid"] = float((df["equity"] - df["cash"] - df["inv_value"]).abs().max())
    return d


def compute_latency_stats(trades: pd.DataFrame) -> Dict:
    out: Dict = {}
    for col, key in [("cancel_latency_s", "cancel"), ("place_latency_s", "place")]:
        if col not in trades.columns:
            continue
        vals = trades[col].dropna().to_numpy(float)
        vals = vals[vals > 1e-6] * 1000.0
        if len(vals) < 2:
            continue
        out[f"{key}_mean_ms"] = float(vals.mean())
        out[f"{key}_p95_ms"]  = float(np.percentile(vals, 95))
        out[f"{key}_p99_ms"]  = float(np.percentile(vals, 99))
        out[f"{key}_n"]       = len(vals)
    return out


# PDF builder :)

def build_pdf(figures: List[plt.Figure]) -> bytes:
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        for fig in figures:
            if fig is not None:
                pdf.savefig(fig, bbox_inches="tight")
    buf.seek(0)
    return buf.read()

# Main

def main() -> None:
    st.set_page_config(
        page_title="MarketSim v3.0 Viewer",
        layout="wide",
    )
    st.title("MarketSim v3.0 — Diagnostic Viewer")
    st.caption(
        "Upload telemetry and trades CSVs produced by market-sim-v3.0.cpp. "
        "Hawkes and replay mode files are both accepted. "
        "multi_seed_summary_v3.csv is optional."
    )

    with st.sidebar:
        st.header("Files")
        tele_file   = st.file_uploader("Telemetry CSV",          type="csv", key="tele")
        trades_file = st.file_uploader("Trades CSV",             type="csv", key="trades")
        seed_file   = st.file_uploader("Multi-seed Summary CSV", type="csv", key="seeds")

        st.divider()
        st.header("Settings")
        log_dt = st.number_input(
            "log_dt (sim seconds per step)",
            value=0.10, min_value=0.001, step=0.01, format="%.3f",
            help="Must match cfg.log_dt in the C++ sim.",
        )
        xcol_choice = st.radio(
            "X-axis for time-series charts",
            ["Simulated time (t column)", "Step index"],
            index=0,
        )
        sharpe_win = st.slider(
            "Rolling IR window (sim seconds)", 5, 120, 30, step=5,
        )
        sample_n = st.slider(
            "Chart downsample factor", 1, 20, 3,
            help="Plot every Nth row to reduce render time on long runs.",
        )
        make_pdf = st.checkbox("Enable PDF export button", value=False)

    if tele_file is None:
        st.info(
            "Upload telemetry_v3_hawkes.csv or telemetry_v3_replay.csv in the sidebar to begin."
        )
        return

    tele_bytes   = tele_file.read()
    trades_bytes = trades_file.read() if trades_file   is not None else None
    seed_bytes   = seed_file.read()   if seed_file     is not None else None

    tele   = load_telemetry(tele_bytes)
    trades = load_trades(trades_bytes)       if trades_bytes is not None else pd.DataFrame()
    seeds  = load_seed_summary(seed_bytes)  if seed_bytes   is not None else pd.DataFrame()

    xcol = "t" if (xcol_choice.startswith("Sim") and "t" in tele.columns) else "step"
    xlab = "Simulated time (s)" if xcol == "t" else "Step index"

    # Detect mode from telemetry columns.
    replay_mode = (
        "hawkes_intensity" in tele.columns
        and tele["hawkes_intensity"].max() < 1.0
    )
    mode_label = "replay" if replay_mode else "hawkes"

    HORIZONS = [1.0, 5.0, 10.0, 30.0]

    mm_fills = pd.DataFrame()
    if not trades.empty and "mm_involved" in trades.columns:
        mm_fills = compute_markout(trades, tele, HORIZONS)
        if not mm_fills.empty:
            mm_fills = enrich_fills_with_tele_state(mm_fills, tele)

    diag     = compute_diagnostics(tele)
    lat_diag = compute_latency_stats(trades) if not trades.empty else {}

    # Top-line metrics 
    fe = diag.get("final_equity",   NaN)
    md = diag.get("max_drawdown",   NaN)
    br = diag.get("bbo_rate",       NaN)
    ar = diag.get("max_acct_resid", NaN)

    c1, c2, c3, c4, c5, c6, c7, c8 = st.columns(8)
    c1.metric("Final equity",   f"${fe:,.3f}" if np.isfinite(fe) else "n/a")
    c2.metric("Max drawdown",   f"${md:,.3f}" if np.isfinite(md) else "n/a")
    c3.metric("Steps",          f"{diag.get('n_steps', 0):,}")
    c4.metric("MM fills",       f"{len(mm_fills):,}")
    c5.metric("BBO valid",      f"{br:.1%}"   if np.isfinite(br) else "n/a")

    acct_ok = np.isfinite(ar) and ar < 1e-4
    c6.metric(
        "Accounting",
        "OK" if acct_ok else (f"FAIL ({ar:.2e})" if np.isfinite(ar) else "n/a"),
    )

    # Latency top-line metrics.
    cmean = lat_diag.get("cancel_mean_ms", NaN)
    cp95  = lat_diag.get("cancel_p95_ms",  NaN)
    c7.metric("Cancel lat (mean)", f"{cmean:.1f} ms" if np.isfinite(cmean) else "n/a")
    c8.metric("Cancel lat (95th)", f"{cp95:.1f} ms"  if np.isfinite(cp95)  else "n/a")

    if np.isfinite(ar) and ar > 1e-4:
        st.error(f"Accounting identity violated: max |equity - (cash + inv_value)| = {ar:.6f}.")
    elif np.isfinite(ar):
        st.success(f"Accounting identity holds (max residual = {ar:.2e}).  Mode: {mode_label}.")

    # Tabs 
    tabs = st.tabs([
        "Overview",
        "Markout Analysis",
        "Adverse Selection",
        "Pricing Signals & Toxicity",
        "LOB Depth",
        "Robustness",
        "Execution & Diagnostics",
    ])

    collected_figs: List[plt.Figure] = []

    # Tab 0: Overview 
    with tabs[0]:
        f = fig_equity_drawdown(tele, xcol, xlab)
        st.pyplot(f); collected_figs.append(f)

        st.divider()
        f = fig_tob_quotes(tele, xcol, xlab, sample_every=sample_n)
        st.pyplot(f); collected_figs.append(f)

        st.divider()
        f = fig_inventory(tele, xcol, xlab)
        st.pyplot(f); collected_figs.append(f)

        st.divider()
        f = fig_pnl_components(tele, xcol, xlab)
        if f:
            st.pyplot(f); collected_figs.append(f)

        st.divider()
        st.subheader("Rolling Information Ratio (simulation-time only)")
        f = fig_rolling_ir(tele, xcol, xlab, sharpe_win, log_dt)
        st.pyplot(f); collected_figs.append(f)
        w_steps = int(round(sharpe_win / log_dt))
        st.caption(
            f"Window = {sharpe_win}s simulated time = {w_steps} steps.  "
            "IR = (mean / std) × √(window_steps).  "
            "**Not annualised to calendar time.**  "
            "The simulation is not a calendar day — do not compare this number to "
            "annualised Sharpe ratios from real strategies."
        )

    # Tab 1: Markout Analysis 
    with tabs[1]:
        if mm_fills.empty:
            st.warning(
                "No MM fills found. Upload a trades CSV and confirm mm_involved column is present."
            )
        else:
            st.subheader("Short-horizon Markout (MM fills only)")
            st.markdown(
                r"""
**Convention:** `markout_H = sign × (fill_px − mid(t + H))`

* `sign = +1`: taker bought from MM ask. Profitable if mid falls after.
* `sign = −1`: taker sold into MM bid. Profitable if mid rises after.

**Positive markout** = fill was profitable at that horizon.
**Negative markout** = adverse selection — filled by an agent with a price edge.
                """
            )
            h_cols_present = [f"markout_{H:.0f}s" for H in HORIZONS
                              if f"markout_{H:.0f}s" in mm_fills.columns]
            if h_cols_present:
                rows = []
                for c in h_cols_present:
                    vals = mm_fills[c].dropna()
                    rows.append({
                        "Horizon": c,
                        "N": len(vals),
                        "Mean ($)":   vals.mean(),
                        "Std ($)":    vals.std(),
                        "Median ($)": vals.median(),
                        "% Adverse":  (vals < 0).mean() * 100 if len(vals) > 0 else NaN,
                    })
                st.dataframe(
                    pd.DataFrame(rows).style.format(
                        {"Mean ($)": "{:.5f}", "Std ($)": "{:.5f}",
                         "Median ($)": "{:.5f}", "% Adverse": "{:.1f}"}
                    ),
                    hide_index=True,
                )

            f = fig_markout_profile(mm_fills, HORIZONS)
            if f:
                st.pyplot(f); collected_figs.append(f)

            f = fig_cumulative_markout(mm_fills, HORIZONS)
            if f:
                st.subheader("Cumulative Markout Over Time")
                st.pyplot(f); collected_figs.append(f)

            f = fig_fill_direction(mm_fills)
            if f:
                st.subheader("Fill Direction Breakdown")
                st.pyplot(f); collected_figs.append(f)

    # Tab 2: Adverse Selection 
    with tabs[2]:
        if mm_fills.empty:
            st.warning("Need a trades CSV with mm_involved column.")
        else:
            st.subheader("Adverse Selection Stratification")
            st.markdown(
                "Each panel groups fills by a structural variable and plots mean markout. "
                "Red = adversely selected; green = profitable.\n\n"
                "**Key questions:**\n"
                "- Does queue position matter? Fills with more queue ahead arrive later, "
                "after more adverse price movement.\n"
                "- Do older orders get worse fills? Stale quotes attract informed flow.\n"
                "- Does heavy queue depletion ahead signal toxicity?\n"
                "- Does book imbalance at fill predict adverse selection direction?"
            )
            f = fig_adverse_selection(mm_fills)
            if f:
                st.pyplot(f); collected_figs.append(f)
            else:
                st.info("Need at least one markout column and structural metadata.")

    # Tab 3: Pricing Signals & Toxicity 
    with tabs[3]:
        st.subheader("Market Maker Pricing Signals")
        st.markdown(
            "The v3.0 MM derives its fair value from **observable book signals only** — "
            "it cannot see the OU fundamental. The reservation price is:\n\n"
            "```\nfair_value  = microprice + imb_coeff × imbalance_ema\n"
            "reservation = fair_value − inv_skew × inventory\n```\n\n"
            "The chart below shows how microprice tracks the BBO mid, and how the "
            "imbalance EMA captures sustained book pressure. These are on separate axes "
            "because they have different units."
        )
        f = fig_pricing_signals(tele, xcol, xlab)
        if f:
            st.pyplot(f); collected_figs.append(f)
        else:
            st.info("microprice or fv_imb_ema not found. Confirm you are using v3.0 telemetry.")

        st.divider()
        st.subheader("Toxicity Signal and Adaptive Spread")
        f = fig_toxicity(tele, xcol, xlab)
        if f:
            st.pyplot(f); collected_figs.append(f)
        else:
            st.info("fill_sign_ema and tox_widen not found in telemetry.")

        f = fig_tox_vs_markout(mm_fills)
        if f:
            st.subheader("Toxicity Widen vs Subsequent Markout")
            st.pyplot(f); collected_figs.append(f)

        if "spread" in tele.columns:
            st.subheader("Quoted Spread Over Time")
            fig_sp, ax_sp = plt.subplots(figsize=(12, 3))
            ax_sp.plot(tele[xcol], tele["spread"], lw=0.8, color="#F57C00", alpha=0.75)
            ax_sp.set_xlabel(xlab)
            ax_sp.set_ylabel("Spread ($)")
            ax_sp.set_title("BBO Spread")
            fig_sp.tight_layout()
            st.pyplot(fig_sp); collected_figs.append(fig_sp)

    # Tab 4: LOB Depth 
    with tabs[4]:
        st.subheader("Order Book Depth Evolution")
        f = fig_lob_depth(tele, xcol, xlab, sample_every=sample_n)
        if f:
            st.pyplot(f); collected_figs.append(f)
        else:
            st.info("bid_l1..ask_l5 columns not found. Confirm you are using v3.0 telemetry.")

        f = fig_imbalance(tele, xcol, xlab)
        if f:
            st.subheader("TOB Imbalance")
            st.pyplot(f); collected_figs.append(f)

        f = fig_tob_share(tele, xcol, xlab)
        if f:
            st.subheader("MM Queue Share at TOB")
            st.pyplot(f); collected_figs.append(f)

    # Tab 5: Robustness 
    with tabs[5]:
        if seeds.empty:
            st.info(
                "Upload multi_seed_summary_v3.csv to see cross-seed robustness. "
                "This file is written automatically by the multi-seed loop in main()."
            )
        else:
            st.subheader("Multi-seed Robustness")
            st.markdown(
                "A strategy that is profitable on only one seed is not robust. "
                "Consistent positive equity across the full sweep is the bar."
            )
            if "final_equity" in seeds.columns:
                fe_all = seeds["final_equity"]
                rc1, rc2, rc3, rc4 = st.columns(4)
                rc1.metric("Seeds positive", f"{int((fe_all > 0).sum())} / {len(fe_all)}")
                rc2.metric("Mean final equity", f"${fe_all.mean():.3f}")
                rc3.metric("Std final equity",  f"${fe_all.std():.3f}")
                rc4.metric("Min final equity",  f"${fe_all.min():.3f}")

            for f in fig_multi_seed(seeds):
                st.pyplot(f); collected_figs.append(f)

            st.subheader("Full Seed Summary")
            fmt_cols = {c: "${:,.4f}" for c in
                        ["final_equity", "max_drawdown", "realized_pnl", "fees_paid"]
                        if c in seeds.columns}
            st.dataframe(seeds.style.format(fmt_cols), hide_index=True)

    # Tab 6: Execution & Diagnostics 
    with tabs[6]:
        st.subheader("Latency Distribution")
        st.markdown(
            "Cancel and place latency are sampled from a log-normal distribution. "
            "The x-axis is **log-scaled** — this is not optional. On a linear axis, "
            "the heavy right tail (the jitter events that cause stale-quote fills) "
            "is compressed into a thin sliver and becomes invisible.\n\n"
            "The gap between the **median** and the **95th/99th percentile** is the "
            "operationally important number: it tells you how often your cancel "
            "arrives materially late."
        )
        f = fig_latency_distribution(trades)
        if f:
            st.pyplot(f); collected_figs.append(f)

            if lat_diag:
                rows_lat = []
                for lat_key, label in [("cancel", "Cancel"), ("place", "Place")]:
                    if f"{lat_key}_n" not in lat_diag:
                        continue
                    rows_lat.append({
                        "Type":     label,
                        "N":        lat_diag[f"{lat_key}_n"],
                        "Mean (ms)": f"{lat_diag[f'{lat_key}_mean_ms']:.2f}",
                        "95th (ms)": f"{lat_diag[f'{lat_key}_p95_ms']:.2f}",
                        "99th (ms)": f"{lat_diag[f'{lat_key}_p99_ms']:.2f}",
                    })
                if rows_lat:
                    st.dataframe(pd.DataFrame(rows_lat), hide_index=True)
        else:
            st.info(
                "No cancel_latency_s / place_latency_s columns found. "
                "Confirm you are using v3.0 trades CSV."
            )

        st.divider()
        st.subheader("Hawkes Flow Intensity")
        f = fig_hawkes_intensity(tele, xcol, xlab)
        if f:
            st.pyplot(f); collected_figs.append(f)
            f2 = fig_latency_vs_intensity(trades, tele)
            if f2:
                st.subheader("Cancel Latency vs Flow Intensity")
                st.markdown(
                    "When `congestion_per_event_s > 0` in the C++ config, high-intensity "
                    "periods add a proportional delay to cancel/place latency. "
                    "A flat scatter here means congestion is disabled (default). "
                    "An upward slope proves the mechanism is working."
                )
                st.pyplot(f2); collected_figs.append(f2)
        else:
            st.info(
                "Hawkes intensity is zero throughout — this is replay mode, "
                "where the generative process is not running."
            )

        st.divider()
        st.subheader("Accounting Identity Verification")
        f = fig_accounting_check(tele, xcol, xlab)
        if f:
            st.pyplot(f); collected_figs.append(f)
        else:
            st.info("Requires equity, cash, and inv_value columns in telemetry.")

        st.subheader("Per-step P&L Distribution")
        f = fig_pnl_hist_qq(tele)
        if f:
            st.pyplot(f); collected_figs.append(f)

        st.subheader("Summary Statistics")
        rows_stats = [
            ("Total steps",            f"{diag.get('n_steps', 0):,}"),
            ("Non-zero P&L steps",     f"{diag.get('n_nonzero', 0):,}"),
            ("Mean step P&L",          f"${diag.get('mean_pnl', NaN):.6f}"),
            ("Std step P&L",           f"${diag.get('std_pnl', NaN):.6f}"),
            ("Median step P&L",        f"${diag.get('median_pnl', NaN):.6f}"),
            ("MAD step P&L",           f"${diag.get('mad_pnl', NaN):.6f}"),
            ("Final equity",           f"${diag.get('final_equity', NaN):,.5f}"),
            ("Max drawdown",           f"${diag.get('max_drawdown', NaN):,.5f}"),
            ("BBO valid rate",         f"{diag.get('bbo_rate', NaN):.2%}"),
            ("Accounting residual",    f"{diag.get('max_acct_resid', NaN):.2e}"),
            ("Sim mode",               mode_label),
        ]
        st.dataframe(
            pd.DataFrame(rows_stats, columns=["Metric", "Value"]),
            hide_index=True,
        )

        with st.expander("Quick diagnostics snippet (copy into notebook)"):
            st.code(
                """\
import pandas as pd
import numpy as np

tele   = pd.read_csv("telemetry_v3_hawkes.csv")
trades = pd.read_csv("trades_v3_hawkes.csv")

# Mid: BBO when valid, fallback to OU fundamental
bid = pd.to_numeric(tele["bid"], errors="coerce")
ask = pd.to_numeric(tele["ask"], errors="coerce")
tele["mid"] = np.where(
    np.isfinite(bid) & np.isfinite(ask),
    0.5 * (bid + ask),
    tele["mid_mark"],
)

# Accounting identity
tele["acct_resid"] = tele["equity"] - tele["cash"] - tele["inv_value"]
print("Max accounting residual:", tele["acct_resid"].abs().max())

# Markout at 5 simulated seconds
mm = trades[trades["mm_involved"] == 1].copy()
t_ref, m_ref = tele["t"].to_numpy(), tele["mid"].to_numpy()
mm["markout_5s"] = mm["sign"] * (
    mm["px"] - np.interp(mm["t"] + 5, t_ref, m_ref, left=np.nan, right=np.nan)
)
print(mm["markout_5s"].describe())

# Latency distribution (v3.0 only)
lat = trades["cancel_latency_s"].dropna()
lat = lat[lat > 1e-6] * 1000  # ms
print(f"Cancel latency — mean: {lat.mean():.1f} ms, 95th: {lat.quantile(0.95):.1f} ms")
                """,
                language="python",
            )

    # PDF export (SIUUU)
    if make_pdf:
        st.divider()
        if st.button("Build PDF report (all charts)"):
            valid_figs = [f for f in collected_figs if f is not None]
            if not valid_figs:
                st.error("No figures available to export.")
            else:
                pdf_bytes = build_pdf(valid_figs)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                st.download_button(
                    label="Download PDF",
                    data=pdf_bytes,
                    file_name=f"marketsim_v30_{ts}.pdf",
                    mime="application/pdf",
                )


if __name__ == "__main__":
    main()
