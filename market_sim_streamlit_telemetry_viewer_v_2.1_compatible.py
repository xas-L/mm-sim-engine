#!/usr/bin/env python3
"""Streamlit app: visualize market-sim telemetry.

v2.1-compatible update from the initial proj:
- Backward compatible with legacy telemetry columns: step,equity,inventory,bid,ask
- Supports event-time column `t` (if present) and uses it as optional x-axis
- Handles one-sided books (bid/ask missing) by falling back to `mid_mark` when available
- Supports additional v5 telemetry columns (if present):
  cash, realized_pnl, unreal_pnl, fees_paid, spread, microprice,
  bid_missing, ask_missing, mm_at_best_bid, mm_at_best_ask,
  tob_share_bid, tob_share_ask
- Optional upload of trades CSV (e.g., trades_v5.csv) for per-trade metrics
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import scipy.stats as stats

import streamlit as st

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 150,
        "axes.titlesize": 16,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "font.family": "DejaVu Sans",
    }
)


# Data loading 

def _coerce_numeric(df: pd.DataFrame, cols: List[str]) -> None:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")


def _ensure_step(df: pd.DataFrame) -> pd.DataFrame:
    if "step" not in df.columns:
        df = df.copy()
        df["step"] = np.arange(len(df), dtype=float)
    return df


@st.cache_data(show_spinner=False)
def load_telemetry_csv(file_bytes: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(file_bytes))
    df = _ensure_step(df)

    numeric_candidates = [
        # legacy
        "step",
        "equity",
        "inventory",
        "bid",
        "ask",
        # v5-ish
        "t",
        "mid_mark",
        "spread",
        "microprice",
        "cash",
        "realized_pnl",
        "unreal_pnl",
        "fees_paid",
        "bid_missing",
        "ask_missing",
        "mm_at_best_bid",
        "mm_at_best_ask",
        "tob_share_bid",
        "tob_share_ask",
    ]
    _coerce_numeric(df, numeric_candidates)

    if "step" in df.columns and df["step"].notna().any():
        df = df.sort_values("step").reset_index(drop=True)
    elif "t" in df.columns and df["t"].notna().any():
        df = df.sort_values("t").reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    # X-axis helper columns
    df["x_step"] = pd.to_numeric(df["step"], errors="coerce")
    if "t" in df.columns:
        df["x_time"] = pd.to_numeric(df["t"], errors="coerce")
    else:
        df["x_time"] = np.nan

    # Compute mid:
    # - if both bid+ask finite => mid_bbo
    # - else if mid_mark exists => mid_mark
    # - else NaN
    if "bid" in df.columns and "ask" in df.columns:
        bid = df["bid"].to_numpy(dtype=float, copy=False)
        ask = df["ask"].to_numpy(dtype=float, copy=False)
        mid_bbo = np.where(np.isfinite(bid) & np.isfinite(ask), 0.5 * (bid + ask), np.nan)
    else:
        mid_bbo = np.full(len(df), np.nan)

    if "mid_mark" in df.columns:
        mid_mark = df["mid_mark"].to_numpy(dtype=float, copy=False)
        df["mid"] = np.where(np.isfinite(mid_bbo), mid_bbo, mid_mark)
    else:
        df["mid"] = mid_bbo

    # Spread (if missing) from BBO
    if "spread" not in df.columns and "bid" in df.columns and "ask" in df.columns:
        spread = np.where(np.isfinite(bid) & np.isfinite(ask), ask - bid, np.nan)
        df["spread"] = spread

    # Per-step P&L from equity differences
    if "equity" in df.columns:
        eq = df["equity"].to_numpy(dtype=float, copy=False)
        step_pnl = np.empty_like(eq)
        step_pnl[:] = np.nan
        if eq.size > 1:
            step_pnl[1:] = eq[1:] - eq[:-1]
        df["step_pnl"] = step_pnl

        # Drawdown
        eq_sanitized = np.where(np.isfinite(eq), eq, -np.inf)
        equity_peak = np.maximum.accumulate(eq_sanitized)
        equity_peak = np.where(np.isfinite(eq), equity_peak, np.nan)
        df["equity_peak"] = equity_peak
        df["drawdown"] = df["equity_peak"] - df["equity"]
    else:
        df["step_pnl"] = np.nan
        df["equity_peak"] = np.nan
        df["drawdown"] = np.nan

  
    if "bid" in df.columns:
        df["bid_finite"] = np.isfinite(df["bid"].to_numpy(dtype=float, copy=False))
    else:
        df["bid_finite"] = False

    if "ask" in df.columns:
        df["ask_finite"] = np.isfinite(df["ask"].to_numpy(dtype=float, copy=False))
    else:
        df["ask_finite"] = False

    df["bbo_valid"] = df["bid_finite"] & df["ask_finite"]

    return df


@st.cache_data(show_spinner=False)
def load_trades_csv(file_bytes: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(file_bytes))
    # best-effort numeric coercion (don’t assume schema)
    for c in ["step", "t", "qty", "size", "price", "pnl", "trade_pnl", "fee", "fees", "fees_paid", "realized_pnl"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def rolling_sharpe(series: pd.Series, window: int, steps_per_day: int, robust: bool = False) -> pd.Series:
    """Rolling Sharpe on per-step P&L with robust option."""
    x = series.astype(float)

    if robust:
        rolling_median = x.rolling(window=window, min_periods=max(5, window // 5)).median()
        mad = x.rolling(window=window, min_periods=max(5, window // 5)).apply(
            lambda y: np.median(np.abs(y - np.median(y))) * 1.4826 if len(y) > 1 else np.nan
        )
        mu = rolling_median
        sd = mad
    else:
        mu = x.rolling(window=window, min_periods=max(5, window // 5)).mean()
        sd = x.rolling(window=window, min_periods=max(5, window // 5)).std(ddof=1)

    ann = np.sqrt(steps_per_day * 252.0)
    sharpe = mu / sd * ann
    sharpe[sd < 1e-8] = np.nan
    return sharpe


def compute_daily_sharpe(df: pd.DataFrame, steps_per_day: int) -> Dict[str, Any]:
    if "step_pnl" not in df.columns or df["step_pnl"].dropna().empty:
        return {"daily_sharpe": np.nan, "daily_mean": np.nan, "daily_std": np.nan, "n_days": 0}

    if "step" not in df.columns:
        return {"daily_sharpe": np.nan, "daily_mean": np.nan, "daily_std": np.nan, "n_days": 0}

    tmp = df[["step", "step_pnl"]].copy()
    tmp["day"] = (tmp["step"] // steps_per_day)
    daily = tmp.groupby("day")["step_pnl"].sum()

    # Remove days with no activity
    daily = daily[daily != 0]

    if len(daily) < 2:
        return {"daily_sharpe": np.nan, "daily_mean": np.nan, "daily_std": np.nan, "n_days": int(len(daily))}

    mean = daily.mean()
    sd = daily.std(ddof=1)
    daily_sharpe = (mean / sd) * np.sqrt(252) if sd > 1e-8 else np.nan

    return {"daily_sharpe": daily_sharpe, "daily_mean": mean, "daily_std": sd, "n_days": int(len(daily))}


def compute_per_trade_ir_from_trades(trades: pd.DataFrame) -> Dict[str, Any]:
    # Compute per-trade IR from a trades file, best-effort based on available columns
    if trades is None or len(trades) == 0:
        return {"per_trade_ir": np.nan, "mean_per_trade": np.nan, "std_per_trade": np.nan, "n_trades": 0, "source": "none"}

    pnl_col = None
    for c in ["trade_pnl", "pnl", "realized_pnl"]:
        if c in trades.columns:
            pnl_col = c
            break

    if pnl_col is None:
        return {"per_trade_ir": np.nan, "mean_per_trade": np.nan, "std_per_trade": np.nan, "n_trades": 0, "source": "trades(no pnl column)"}

    trade_pnl = pd.to_numeric(trades[pnl_col], errors="coerce").dropna()
    trade_pnl = trade_pnl[np.abs(trade_pnl) > 1e-6]

    if len(trade_pnl) < 2:
        return {"per_trade_ir": np.nan, "mean_per_trade": np.nan, "std_per_trade": np.nan, "n_trades": int(len(trade_pnl)), "source": f"trades({pnl_col})"}

    mean_trade = trade_pnl.mean()
    std_trade = trade_pnl.std(ddof=1)
    per_trade_ir = mean_trade / std_trade if std_trade > 1e-8 else np.nan

    return {
        "per_trade_ir": per_trade_ir,
        "mean_per_trade": mean_trade,
        "std_per_trade": std_trade,
        "n_trades": int(len(trade_pnl)),
        "source": f"trades({pnl_col})",
    }


def compute_per_trade_ir_from_inventory(df: pd.DataFrame) -> Dict[str, Any]:
    # Fallback per-trade IR using inventory changes as proxy for trades
    if "inventory" not in df.columns:
        return {"per_trade_ir": np.nan, "mean_per_trade": np.nan, "std_per_trade": np.nan, "n_trades": 0, "source": "inventory(n/a)"}

    inventory_changes = df["inventory"].diff().fillna(0)
    trade_steps = df.loc[inventory_changes != 0].copy()

    if len(trade_steps) < 2:
        return {"per_trade_ir": np.nan, "mean_per_trade": np.nan, "std_per_trade": np.nan, "n_trades": int(len(trade_steps)), "source": "inventory"}

    trade_steps["trade_pnl"] = trade_steps.get("step_pnl", np.nan)
    trade_pnl = pd.to_numeric(trade_steps["trade_pnl"], errors="coerce").dropna()
    trade_pnl = trade_pnl[np.abs(trade_pnl) > 1e-6]

    if len(trade_pnl) < 2:
        return {"per_trade_ir": np.nan, "mean_per_trade": np.nan, "std_per_trade": np.nan, "n_trades": int(len(trade_pnl)), "source": "inventory"}

    mean_trade = trade_pnl.mean()
    std_trade = trade_pnl.std(ddof=1)
    per_trade_ir = mean_trade / std_trade if std_trade > 1e-8 else np.nan

    return {
        "per_trade_ir": per_trade_ir,
        "mean_per_trade": mean_trade,
        "std_per_trade": std_trade,
        "n_trades": int(len(trade_pnl)),
        "source": "inventory",
    }


def run_diagnostics(df: pd.DataFrame) -> Dict[str, Any]:
    if "step_pnl" not in df.columns:
        return {}

    d = df["step_pnl"].dropna()

    stats_dict: Dict[str, Any] = {
        "total_rows": int(len(df)),
        "non_zero_pnl": int((d != 0).sum()),
        "max_step_pnl": float(d.max()) if len(d) else np.nan,
        "min_step_pnl": float(d.min()) if len(d) else np.nan,
        "mean_step_pnl": float(d.mean()) if len(d) else np.nan,
        "std_step_pnl": float(d.std(ddof=1)) if len(d) > 1 else np.nan,
        "median_step_pnl": float(d.median()) if len(d) else np.nan,
        "mad_step_pnl": float(stats.median_abs_deviation(d, scale="normal")) if len(d) > 1 else np.nan,
        "bbo_valid_rate": float(df.get("bbo_valid", pd.Series(False)).mean()) if len(df) else np.nan,
    }

    if len(d) > 0:
        largest_pos = df.nlargest(5, "step_pnl")[["step", "step_pnl"]].values.tolist()
        largest_neg = df.nsmallest(5, "step_pnl")[["step", "step_pnl"]].values.tolist()
        stats_dict["largest_pos"] = largest_pos
        stats_dict["largest_neg"] = largest_neg

        idx = d.abs().idxmax()
        stats_dict["worst_row_idx"] = int(idx)
        stats_dict["worst_step_pnl"] = float(d.loc[idx]) if idx in d.index else np.nan

        std = d.std(ddof=1)
        if std > 1e-8:
            stats_dict["max_z_score"] = float(abs((d.max() - d.mean()) / std))
            stats_dict["min_z_score"] = float(abs((d.min() - d.mean()) / std))
        else:
            stats_dict["max_z_score"] = np.nan
            stats_dict["min_z_score"] = np.nan

    # Duplicates
    if "step" in df.columns:
        step_counts = df["step"].value_counts()
        stats_dict["duplicate_steps"] = int((step_counts > 1).sum())
    else:
        stats_dict["duplicate_steps"] = 0

    return stats_dict


def check_plausibility(stats_dict: Dict[str, Any]) -> List[str]:
    warnings: List[str] = []

    sharpe = stats_dict.get("sharpe_annualized", np.nan)
    if np.isfinite(sharpe):
        if sharpe > 20:
            warnings.append(f" Attention. Sharpe >20 (value: {sharpe:.2f}) — almost certainly a bug")
        elif sharpe > 10:
            warnings.append(f" Attention. Sharpe >10 (value: {sharpe:.2f}) — review carefully")
        elif sharpe < -5:
            warnings.append(f" Unrealistic Sharpe < -5 (value: {sharpe:.2f}) — extreme negative performance")

        if sharpe < 0.5 or sharpe > 5.0:
            warnings.append(f"Sharpe {sharpe:.2f} outside typical MM range (0.5–5.0)")

    daily_sharpe = stats_dict.get("daily_sharpe", np.nan)
    if np.isfinite(daily_sharpe) and abs(daily_sharpe) > 10:
        warnings.append(f"Daily Sharpe magnitude >10 (value: {daily_sharpe:.2f})")

    max_z = stats_dict.get("max_z_score", np.nan)
    if np.isfinite(max_z) and max_z > 10:
        warnings.append(f"Extreme P&L outlier (z-score: {max_z:.1f})")

    dup = stats_dict.get("duplicate_steps", 0)
    if dup and dup > 0:
        warnings.append(f"Found {dup} duplicate steps — check telemetry")

    bbo_rate = stats_dict.get("bbo_valid_rate", np.nan)
    if np.isfinite(bbo_rate) and bbo_rate < 0.8:
        warnings.append(f"BBO valid rate is low ({bbo_rate:.1%}) — one-sided book is common; ensure mid fallback is used")

    return warnings


# plots

def fig_equity(df: pd.DataFrame, x: str, xlab: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df[x], df.get("equity", np.nan), label="Equity")

    if "equity_peak" in df.columns and "drawdown" in df.columns:
        ax.fill_between(df[x], df.get("equity", np.nan), df.get("equity_peak", np.nan), alpha=0.15, label="Drawdown")
        if df["drawdown"].notna().any():
            idx_max_dd = df["drawdown"].idxmax()
            if pd.notna(idx_max_dd):
                x_dd = df.loc[idx_max_dd, x]
                dd_val = df.loc[idx_max_dd, "drawdown"]
                eq_val = df.loc[idx_max_dd, "equity"] if "equity" in df.columns else np.nan
                if np.isfinite(dd_val) and np.isfinite(eq_val):
                    ax.annotate(
                        f"Max DD: {dd_val:,.2f}",
                        xy=(x_dd, eq_val),
                        xytext=(10, 20),
                        textcoords="offset points",
                        arrowprops=dict(arrowstyle="->", lw=1),
                    )

    ax.set_title("Equity Curve (with drawdown shading)")
    ax.set_xlabel(xlab)
    ax.set_ylabel("Equity ($)")
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def fig_drawdown(df: pd.DataFrame, x: str, xlab: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(df[x], df.get("drawdown", np.nan), label="Drawdown")
    ax.set_title("Drawdown vs Time")
    ax.set_xlabel(xlab)
    ax.set_ylabel("Drawdown ($)")
    fig.tight_layout()
    return fig


def fig_inventory(df: pd.DataFrame, x: str, xlab: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 3))
    if "inventory" in df.columns:
        ax.step(df[x], df["inventory"], where="post", label="Inventory")
    ax.set_title("Inventory over Time")
    ax.set_xlabel(xlab)
    ax.set_ylabel("Shares")
    fig.tight_layout()
    return fig


def fig_market_micro(df: pd.DataFrame, x: str, xlab: str, sample_every: int = 10) -> plt.Figure:
    d = df.iloc[:: max(1, sample_every)].copy()
    fig, ax = plt.subplots(figsize=(10, 3.5))

    if "bid" in d.columns:
        ax.plot(d[x], d["bid"], lw=1, alpha=0.9, label="Bid")
    if "ask" in d.columns:
        ax.plot(d[x], d["ask"], lw=1, alpha=0.9, label="Ask")
    if "mid" in d.columns and d["mid"].notna().any():
        ax.plot(d[x], d["mid"], lw=1, ls="--", alpha=0.8, label="Mid")

    ax.set_title("Top of Book (bid/ask/mid)")
    ax.set_xlabel(xlab)
    ax.set_ylabel("Price")
    ax.legend(loc="best", ncol=3)
    fig.tight_layout()
    return fig


def fig_spread_and_microprice(df: pd.DataFrame, x: str, xlab: str, sample_every: int = 10) -> Optional[plt.Figure]:
    cols_present = [c for c in ["spread", "microprice"] if c in df.columns and df[c].notna().any()]
    if not cols_present:
        return None

    d = df.iloc[:: max(1, sample_every)].copy()
    fig, ax = plt.subplots(figsize=(10, 3.5))

    if "spread" in cols_present:
        ax.plot(d[x], d["spread"], lw=1.2, label="Spread")
    if "microprice" in cols_present:
        ax.plot(d[x], d["microprice"], lw=1.2, label="Microprice")

    ax.set_title("Spread / Microprice")
    ax.set_xlabel(xlab)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def fig_pnl_components(df: pd.DataFrame, x: str, xlab: str, sample_every: int = 5) -> Optional[plt.Figure]:
    cols = [c for c in ["realized_pnl", "unreal_pnl", "fees_paid", "cash"] if c in df.columns and df[c].notna().any()]
    if not cols:
        return None

    d = df.iloc[:: max(1, sample_every)].copy()
    fig, ax = plt.subplots(figsize=(10, 3.5))

    for c in cols:
        ax.plot(d[x], d[c], lw=1.2, label=c)

    ax.set_title("P&L Components / Cash")
    ax.set_xlabel(xlab)
    ax.legend(loc="best", ncol=2)
    fig.tight_layout()
    return fig


def fig_step_pnl_hist(df: pd.DataFrame, bins: int = 60) -> plt.Figure:
    d = df.get("step_pnl", pd.Series(dtype=float)).dropna()
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(d, bins=bins, kde=True, ax=ax)
    ax.set_title("Per-step P&L distribution")
    ax.set_xlabel("Δ Equity per step ($)")
    ax.set_ylabel("Count")
    fig.tight_layout()
    return fig


def fig_rolling_sharpe(df: pd.DataFrame, x: str, xlab: str, window: int, steps_per_day: int, robust: bool = False) -> plt.Figure:
    rs = rolling_sharpe(df.get("step_pnl", pd.Series(dtype=float)), window=window, steps_per_day=steps_per_day, robust=robust)
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(df[x], rs, lw=1.2, label="Robust Sharpe" if robust else "Traditional Sharpe")
    ax.axhline(0.0, lw=1, ls=":")
    ax.set_title(f"Rolling Sharpe (window={window} steps, {'robust' if robust else 'traditional'})")
    ax.set_xlabel(xlab)
    ax.set_ylabel("Sharpe (annualized)")
    ax.set_ylim(-30, 30)
    fig.tight_layout()
    return fig


def fig_inventory_vs_step_pnl(df: pd.DataFrame, horizon: int = 1) -> plt.Figure:
    inv = pd.to_numeric(df.get("inventory", np.nan), errors="coerce").to_numpy()
    pnl = pd.to_numeric(df.get("step_pnl", np.nan), errors="coerce").to_numpy()

    if len(inv) < horizon + 1:
        horizon = 1

    x = inv[: -horizon]
    y = pnl[horizon:]
    mask = np.isfinite(x) & np.isfinite(y)

    fig, ax = plt.subplots(figsize=(6.5, 5))
    sns.regplot(x=x[mask], y=y[mask], ax=ax, scatter_kws={"s": 12, "alpha": 0.6}, line_kws={"lw": 2})
    ax.set_title(f"Inventory vs next-step P&L (h={horizon})")
    ax.set_xlabel("Inventory (t)")
    ax.set_ylabel("Δ Equity (t+h)")
    fig.tight_layout()
    return fig


def fig_diagnostics(df: pd.DataFrame, x: str, xlab: str, steps_per_day: int) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    step_pnl = df.get("step_pnl", pd.Series(dtype=float)).dropna()

    # 1) Per-step P&L, outliers highlighted
    ax = axes[0, 0]
    if len(step_pnl):
        ax.plot(df.loc[step_pnl.index, x], step_pnl.values, alpha=0.6)
        if len(step_pnl) >= 10:
            largest_pos = step_pnl.nlargest(3)
            largest_neg = step_pnl.nsmallest(3)
            ax.scatter(df.loc[largest_pos.index, x], largest_pos.values, s=50, zorder=5, label="Top 3 gains")
            ax.scatter(df.loc[largest_neg.index, x], largest_neg.values, s=50, zorder=5, label="Top 3 losses")
            ax.legend()
    else:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center")

    ax.set_title("Per-step P&L with outliers")
    ax.set_xlabel(xlab)
    ax.set_ylabel("P&L ($)")

    # 2) Histogram, log scale
    ax = axes[0, 1]
    if len(step_pnl):
        ax.hist(step_pnl, bins=101, edgecolor="black")
        ax.set_yscale("log")
        ax.set_title("P&L histogram (log scale)")
        ax.set_xlabel("P&L ($)")
        ax.set_ylabel("Count (log)")
    else:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center")
        ax.set_title("P&L histogram")

    # Easter egg 1-0 Will Grigg up the Tics!
    ax = axes[1, 0]
    if len(step_pnl) > 10:
        stats.probplot(step_pnl, dist="norm", plot=ax)
        ax.set_title("Q-Q plot vs Normal")
    else:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center")
        ax.set_title("Q-Q plot")

    # 4) Rolling vol
    ax = axes[1, 1]
    window = min(200, max(0, len(step_pnl) // 4))
    if window > 10:
        rolling_vol = step_pnl.rolling(window=window).std(ddof=1) * np.sqrt(steps_per_day * 252)
        ax.plot(df.loc[rolling_vol.index, x], rolling_vol.values)
        ax.set_title(f"Rolling annualized volatility (window={window})")
        ax.set_xlabel(xlab)
        ax.set_ylabel("Volatility (annualized)")
    else:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center")
        ax.set_title("Rolling volatility")

    fig.tight_layout()
    return fig


# pdf 

def build_summary_pdf(figs: List[plt.Figure]) -> bytes:
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        for fig in figs:
            pdf.savefig(fig, bbox_inches="tight")
    buf.seek(0)
    return buf.read()


# streamlit app

def main() -> None:
    st.set_page_config(page_title="MarketSim Telemetry", layout="wide")
    st.title("Market Simulation Telemetry Viewer")

    with st.sidebar:
        st.header("Controls")

        theme = st.radio("Theme", ["Light", "Dark"], horizontal=True)
        if theme == "Dark":
            plt.style.use("dark_background")
        else:
            plt.style.use("default")

        uploaded = st.file_uploader("Telemetry CSV", type=["csv"], accept_multiple_files=False)
        uploaded_trades = st.file_uploader("(Optional) Trades CSV", type=["csv"], accept_multiple_files=False)

        default_path = st.text_input("…or path to telemetry CSV on disk", value="telemetry.csv")
        default_trades_path = st.text_input("…or path to trades CSV on disk", value="trades_v5.csv")

        st.subheader("Sharpe Computation")
        window = st.slider("Rolling Sharpe window (steps)", min_value=50, max_value=2000, value=200, step=10)
        steps_per_day = st.number_input("Steps per day (annualization)", min_value=1, max_value=2000, value=390)
        robust_sharpe = st.checkbox("Use robust Sharpe (Median/MAD)", value=True)

        st.subheader("X-axis")
        x_mode = st.radio("Use", ["Step", "Event time (t)"], index=0, help="If `t` is missing, the app falls back to Step.")

        st.subheader("Plots")
        show = {
            "equity": st.checkbox("Equity curve", True),
            "drawdown": st.checkbox("Drawdown", True),
            "inventory": st.checkbox("Inventory", True),
            "micro": st.checkbox("Top of Book", True),
            "spread_micro": st.checkbox("Spread / Microprice", True),
            "pnl_components": st.checkbox("P&L components / Cash", True),
            "hist": st.checkbox("P&L histogram", True),
            "rsharpe": st.checkbox("Rolling Sharpe", True),
            "rsharpe_robust": st.checkbox("Robust Rolling Sharpe", True),
            "inv_vs_pnl": st.checkbox("Inventory vs P&L", True),
            "diagnostics": st.checkbox("Diagnostics", True),
        }

        sample_every = st.slider("Downsample factor", min_value=1, max_value=50, value=10)
        make_pdf = st.checkbox("Build PDF report", value=False)

        if st.button("Generate demo data"):
            steps = np.arange(1000)
            rng = np.random.default_rng(123)
            equity = 10000 + np.cumsum(rng.standard_normal(1000) * 5)
            inventory = rng.integers(-5, 6, 1000)
            t = np.cumsum(np.maximum(0.0, rng.standard_exponential(1.0, 1000) - 0.25))

            mid = 100 + np.cumsum(rng.standard_normal(1000) * 0.01)
            bid = mid - 0.01
            ask = mid + 0.01

            df_demo = pd.DataFrame({"step": steps, "t": t, "equity": equity, "inventory": inventory, "bid": bid, "ask": ask})
            df_demo["mid_mark"] = mid
            df_demo["cash"] = 100000 + np.cumsum(rng.standard_normal(1000) * 2)
            df_demo["realized_pnl"] = np.cumsum(rng.standard_normal(1000) * 0.5)
            df_demo["unreal_pnl"] = rng.standard_normal(1000) * 3
            df_demo["fees_paid"] = np.cumsum(np.abs(rng.standard_normal(1000) * 0.01))
            df_demo["spread"] = (df_demo["ask"] - df_demo["bid"]) + np.abs(rng.standard_normal(1000) * 0.001)
            st.session_state["demo_df"] = df_demo

    # Load telemetry
    df: Optional[pd.DataFrame] = None
    trades_df: Optional[pd.DataFrame] = None

    if "demo_df" in st.session_state:
        df = st.session_state["demo_df"].copy()
        st.info("📊 Displaying demo data")
    elif uploaded is not None:
        try:
            df = load_telemetry_csv(uploaded.read())
            st.success("✅ Telemetry CSV loaded successfully")
        except Exception as e:
            st.error(f"Failed to parse uploaded telemetry CSV: {e}")
    else:
        path = Path(default_path)
        if path.exists():
            try:
                df = load_telemetry_csv(path.read_bytes())
                st.success(f"✅ Loaded {path.name}")
            except Exception as e:
                st.error(f"Failed to parse telemetry CSV at {path}: {e}")
        else:
            st.info("No telemetry file uploaded and default path not found. Generate demo data or upload a CSV.")

    if df is None:
        st.stop()
    # Load trades (optional)
    if uploaded_trades is not None:
        try:
            trades_df = load_trades_csv(uploaded_trades.read())
            st.success("✅ Trades CSV loaded")
        except Exception as e:
            st.warning(f"Trades CSV failed to load: {e}")
    else:
        tpath = Path(default_trades_path)
        if tpath.exists():
            try:
                trades_df = load_trades_csv(tpath.read_bytes())
                st.info(f"Loaded trades file: {tpath.name}")
            except Exception:
                trades_df = None


    if x_mode == "Event time (t)" and "x_time" in df.columns and df["x_time"].notna().any():
        xcol, xlab = "x_time", "Event time (t)"
    else:
        xcol, xlab = "x_step", "Step"


    # Sharpe
    step_pnl = pd.to_numeric(df.get("step_pnl", np.nan), errors="coerce").dropna().to_numpy()
    if step_pnl.size > 1 and np.nanstd(step_pnl, ddof=1) > 1e-8:
        sharpe_per_step = np.nanmean(step_pnl) / np.nanstd(step_pnl, ddof=1) * np.sqrt(steps_per_day * 252.0)
    else:
        sharpe_per_step = np.nan

    daily_sharpe_dict = compute_daily_sharpe(df, steps_per_day)

    per_trade_dict = compute_per_trade_ir_from_trades(trades_df) if trades_df is not None else {"source": "none"}
    if not np.isfinite(per_trade_dict.get("per_trade_ir", np.nan)):
        per_trade_dict = compute_per_trade_ir_from_inventory(df)

    diagnostics = run_diagnostics(df)

    plausibility_warnings = check_plausibility(
        {
            "sharpe_annualized": sharpe_per_step,
            "daily_sharpe": daily_sharpe_dict.get("daily_sharpe", np.nan),
            **diagnostics,
        }
    )

    # KPIs

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        final_eq = df["equity"].dropna().iloc[-1] if "equity" in df.columns and df["equity"].notna().any() else np.nan
        st.metric("Final equity", f"${final_eq:,.2f}" if np.isfinite(final_eq) else "n/a")

    with col2:
        max_dd = df["drawdown"].max(skipna=True) if "drawdown" in df.columns else np.nan
        st.metric("Max drawdown", f"${max_dd:,.2f}" if np.isfinite(max_dd) else "n/a")

    with col3:
        st.metric("Sharpe (per-step)", f"{sharpe_per_step:,.2f}" if np.isfinite(sharpe_per_step) else "n/a")

    with col4:
        daily_sharpe_val = daily_sharpe_dict.get("daily_sharpe", np.nan)
        st.metric("Daily Sharpe", f"{daily_sharpe_val:,.2f}" if np.isfinite(daily_sharpe_val) else "n/a")

    with col5:
        per_trade_ir_val = per_trade_dict.get("per_trade_ir", np.nan)
        src = per_trade_dict.get("source", "")
        st.metric("Per-trade IR", f"{per_trade_ir_val:,.2f}" if np.isfinite(per_trade_ir_val) else "n/a", help=f"Source: {src}")

    # Warnings 

    if plausibility_warnings:
        with st.expander("Plausibility Warnings!", expanded=True):
            for warning in plausibility_warnings:
                st.warning(warning)

    # ==================== ROBUST METRICS ====================

    with st.expander("📈 Robust Performance Metrics", expanded=True):
        cols = st.columns(4)
        with cols[0]:
            st.metric("Rows", f"{len(df):,}")
        with cols[1]:
            st.metric("Non-zero P&L steps", f"{diagnostics.get('non_zero_pnl', 0):,}")
        with cols[2]:
            median_pnl = diagnostics.get("median_step_pnl", np.nan)
            st.metric("Median step P&L", f"${median_pnl:.2f}" if np.isfinite(median_pnl) else "n/a")
        with cols[3]:
            mad_pnl = diagnostics.get("mad_step_pnl", np.nan)
            st.metric("MAD step P&L", f"${mad_pnl:.2f}" if np.isfinite(mad_pnl) else "n/a")

        st.subheader("Daily Aggregation")
        daily_cols = st.columns(4)
        with daily_cols[0]:
            st.metric("Days", f"{daily_sharpe_dict.get('n_days', 0):,}")
        with daily_cols[1]:
            daily_mean = daily_sharpe_dict.get("daily_mean", np.nan)
            st.metric("Daily mean P&L", f"${daily_mean:.2f}" if np.isfinite(daily_mean) else "n/a")
        with daily_cols[2]:
            daily_std = daily_sharpe_dict.get("daily_std", np.nan)
            st.metric("Daily std", f"${daily_std:.2f}" if np.isfinite(daily_std) else "n/a")
        with daily_cols[3]:
            st.metric("Trades", f"{per_trade_dict.get('n_trades', 0):,}")

        bbo_rate = diagnostics.get("bbo_valid_rate", np.nan)
        if np.isfinite(bbo_rate):
            st.caption(f"BBO valid rate: {bbo_rate:.1%} (mid falls back to mid_mark when BBO is one-sided)")

    # drill down

    with st.expander(" Diagnostics & Outlier Detection", expanded=False):
        if diagnostics:
            st.subheader("Largest P&L Movers")
            if diagnostics.get("largest_pos"):
                pos_df = pd.DataFrame(diagnostics["largest_pos"], columns=["Step", "P&L"])
                st.write("**Top 5 gains:**")
                st.dataframe(pos_df.style.format({"P&L": "${:,.2f}"}))
            if diagnostics.get("largest_neg"):
                neg_df = pd.DataFrame(diagnostics["largest_neg"], columns=["Step", "P&L"])
                st.write("**Top 5 losses:**")
                st.dataframe(neg_df.style.format({"P&L": "${:,.2f}"}))

            st.subheader("Basic Statistics")
            stats_df = pd.DataFrame(
                [
                    {"Metric": "Rows", "Value": diagnostics.get("total_rows", 0)},
                    {"Metric": "Non-zero P&L steps", "Value": diagnostics.get("non_zero_pnl", 0)},
                    {"Metric": "Mean step P&L", "Value": f"${diagnostics.get('mean_step_pnl', np.nan):.2f}"},
                    {"Metric": "Std step P&L", "Value": f"${diagnostics.get('std_step_pnl', np.nan):.2f}"},
                    {"Metric": "Median step P&L", "Value": f"${diagnostics.get('median_step_pnl', np.nan):.2f}"},
                    {"Metric": "MAD step P&L", "Value": f"${diagnostics.get('mad_step_pnl', np.nan):.2f}"},
                    {"Metric": "Max Z-score", "Value": f"{diagnostics.get('max_z_score', np.nan):.1f}" if np.isfinite(diagnostics.get("max_z_score", np.nan)) else "n/a"},
                    {"Metric": "Duplicate steps", "Value": diagnostics.get("duplicate_steps", 0)},
                    {"Metric": "BBO valid rate", "Value": f"{diagnostics.get('bbo_valid_rate', np.nan):.1%}" if np.isfinite(diagnostics.get("bbo_valid_rate", np.nan)) else "n/a"},
                ]
            )
            st.dataframe(stats_df, hide_index=True)

            if "worst_row_idx" in diagnostics:
                idx = diagnostics["worst_row_idx"]
                st.write(f"**Worst single step (row index {idx}):**")
                if idx in df.index:
                    context = df.loc[max(0, idx - 2) : idx + 2]
                    fmt = {}
                    for c in ["equity", "step_pnl", "cash", "realized_pnl", "unreal_pnl", "fees_paid"]:
                        if c in context.columns:
                            fmt[c] = "${:,.2f}"
                    for c in ["bid", "ask", "mid", "mid_mark", "spread", "microprice"]:
                        if c in context.columns:
                            fmt[c] = "{:.6f}"
                    st.dataframe(context.style.format(fmt))

        if show["diagnostics"]:
            st.pyplot(fig_diagnostics(df, x=xcol, xlab=xlab, steps_per_day=steps_per_day))

    # chart items
    figs: List[plt.Figure] = []

    if show["equity"]:
        with st.expander("Equity curve", expanded=True):
            fig = fig_equity(df, x=xcol, xlab=xlab)
            st.pyplot(fig)
            figs.append(fig)

    if show["drawdown"]:
        with st.expander("Drawdown", expanded=False):
            fig = fig_drawdown(df, x=xcol, xlab=xlab)
            st.pyplot(fig)
            figs.append(fig)

    if show["inventory"]:
        with st.expander("Inventory", expanded=False):
            fig = fig_inventory(df, x=xcol, xlab=xlab)
            st.pyplot(fig)
            figs.append(fig)

    if show["micro"]:
        with st.expander("Top of Book (bid/ask/mid)", expanded=False):
            fig = fig_market_micro(df, x=xcol, xlab=xlab, sample_every=sample_every)
            st.pyplot(fig)
            figs.append(fig)

    if show["spread_micro"]:
        fig = fig_spread_and_microprice(df, x=xcol, xlab=xlab, sample_every=sample_every)
        if fig is not None:
            with st.expander("Spread / Microprice", expanded=False):
                st.pyplot(fig)
                figs.append(fig)

    if show["pnl_components"]:
        fig = fig_pnl_components(df, x=xcol, xlab=xlab, sample_every=max(1, sample_every // 2))
        if fig is not None:
            with st.expander("P&L components / Cash", expanded=False):
                st.pyplot(fig)
                figs.append(fig)

    if show["hist"]:
        with st.expander("Per-step P&L distribution", expanded=False):
            fig = fig_step_pnl_hist(df)
            st.pyplot(fig)
            figs.append(fig)

    if show["rsharpe"]:
        with st.expander("Rolling Sharpe (traditional)", expanded=False):
            fig = fig_rolling_sharpe(df, x=xcol, xlab=xlab, window=window, steps_per_day=steps_per_day, robust=False)
            st.pyplot(fig)
            figs.append(fig)

    if show["rsharpe_robust"]:
        with st.expander("Rolling Sharpe (robust)", expanded=False):
            fig = fig_rolling_sharpe(df, x=xcol, xlab=xlab, window=window, steps_per_day=steps_per_day, robust=True)
            st.pyplot(fig)
            figs.append(fig)

    if show["inv_vs_pnl"]:
        with st.expander("Inventory vs next-step P&L", expanded=False):
            fig = fig_inventory_vs_step_pnl(df, horizon=1)
            st.pyplot(fig)
            figs.append(fig)

    # export

    if make_pdf:
        if st.button("📄 Build & download comprehensive PDF report", type="primary"):
            try:
                pdf_bytes = build_summary_pdf(figs)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"marketsim_report_{timestamp}.pdf"
                st.download_button("Download PDF report", data=pdf_bytes, file_name=filename, mime="application/pdf")
            except Exception as e:
                st.error(f"Failed to build PDF: {e}")

    with st.expander("📋 Quick Diagnostics Code", expanded=False):
        st.code(
            """
import pandas as pd
import numpy as np

# Telemetry
tele = pd.read_csv('telemetry.csv')
if 'step' not in tele.columns:
    tele['step'] = np.arange(len(tele))
tele = tele.sort_values('step').reset_index(drop=True)

# Mid with one-sided fallback
bid = pd.to_numeric(tele.get('bid'), errors='coerce')
ask = pd.to_numeric(tele.get('ask'), errors='coerce')
mid_bbo = np.where(np.isfinite(bid) & np.isfinite(ask), 0.5*(bid+ask), np.nan)
if 'mid_mark' in tele.columns:
    mid_mark = pd.to_numeric(tele['mid_mark'], errors='coerce')
    tele['mid'] = np.where(np.isfinite(mid_bbo), mid_bbo, mid_mark)
else:
    tele['mid'] = mid_bbo

# Per-step pnl
tele['equity'] = pd.to_numeric(tele.get('equity'), errors='coerce')
tele['step_pnl'] = tele['equity'].diff()

# Sharpe sanity check
steps_per_day = 390
ann = np.sqrt(steps_per_day * 252.0)
d = tele['step_pnl'].dropna()
sh = (d.mean()/d.std(ddof=1))*ann if len(d)>1 and d.std(ddof=1)>1e-8 else np.nan
print('sharpe:', sh)

# Biggest spikes
idx = tele['step_pnl'].abs().idxmax()
print('worst row:', idx)
print(tele.loc[max(0, idx-5):idx+5])
            """,
            language="python",
        )


if __name__ == "__main__":
    main()
