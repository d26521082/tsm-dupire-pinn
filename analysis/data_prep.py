"""Clean features_all.csv into an OTM-only unified call-equivalent dataset.

Steps:
  1. Filter options priced below intrinsic value
  2. Liquidity filter (drop market_price < threshold)
  3. Estimate implied dividend yield q from put-call parity
  4. Build OTM-only unified surface: K>S use call; K<S use put converted to call
  5. Invert Black-Scholes (with dividend) for per-option implied volatility
  6. Save cleaned data and a summary plot
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "features_all.csv"
OUT_DIR = ROOT / "data"
OUT_DIR.mkdir(exist_ok=True)
FIG_DIR = ROOT / "figures" / "data_prep"
FIG_DIR.mkdir(parents=True, exist_ok=True)

LIQ_THRESHOLD = 0.05  # drop quotes below this price (deep-OTM noise)
INTRINSIC_TOL = 0.01  # tolerance for below-intrinsic check


def bs_price(S: float, K: float, T: float, r: float, q: float, sigma: float, opt_type: str) -> float:
    if T <= 0 or sigma <= 0:
        return max(0.0, (S - K) if opt_type == "call" else (K - S))
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if opt_type == "call":
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def iv_invert(price: float, S: float, K: float, T: float, r: float, q: float, opt_type: str) -> float:
    if T <= 0:
        return np.nan
    intrinsic = max(0.0, (S - K) if opt_type == "call" else (K - S))
    if price < intrinsic - 1e-6:
        return np.nan
    try:
        return brentq(
            lambda s: bs_price(S, K, T, r, q, s, opt_type) - price,
            1e-6, 5.0, xtol=1e-8, maxiter=200,
        )
    except (ValueError, RuntimeError):
        return np.nan


def section(title: str) -> None:
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def main() -> None:
    df = pd.read_csv(CSV)
    n0 = len(df)
    print(f"loaded: {n0} rows")

    # ---- 1. below-intrinsic filter ----
    intrinsic = np.where(
        df.optionType == "call",
        np.maximum(df["S"] - df["K"], 0),
        np.maximum(df["K"] - df["S"], 0),
    )
    keep = df.market_price >= intrinsic - INTRINSIC_TOL
    df = df[keep].reset_index(drop=True)
    n1 = len(df)
    print(f"after below-intrinsic filter: {n1}  (dropped {n0 - n1})")

    # ---- 2. liquidity filter ----
    df = df[df.market_price >= LIQ_THRESHOLD].reset_index(drop=True)
    n2 = len(df)
    print(f"after liquidity filter (>= ${LIQ_THRESHOLD}): {n2}  (dropped {n1 - n2})")

    # ---- 3. implied dividend yield from put-call parity ----
    # C - P = S e^{-qT} - K e^{-rT}  =>  q = -ln((C - P + K e^{-rT}) / S) / T
    section("[3] Implied dividend yield (from put-call parity)")
    parities = []
    for (exp, K), g in df.groupby(["expirationDate", "K"]):
        c = g[g.optionType == "call"]
        p = g[g.optionType == "put"]
        if len(c) == 1 and len(p) == 1:
            S = float(c["S"].iloc[0])
            T = float(c["T"].iloc[0])
            r = float(c["r"].iloc[0])
            C = float(c.market_price.iloc[0])
            P = float(p.market_price.iloc[0])
            num = C - P + K * np.exp(-r * T)
            if num > 0 and T > 0:
                q = -np.log(num / S) / T
                parities.append({"exp": exp, "K": K, "T": T, "q": q})
    par = pd.DataFrame(parities)
    print(f"matched (K,T) pairs: {len(par)}")

    # robust: use median q across pairs with T >= 0.05 (short-T q is noisy)
    par_robust = par[par["T"] >= 0.05]
    q_implied = float(par_robust["q"].median())
    print(
        f"q (median, T>=0.05): {q_implied:.6f}  ({q_implied * 100:.3f}%)"
    )
    print(
        f"q stats (all pairs):  median={par['q'].median():.4f}  "
        f"mean={par['q'].mean():.4f}  std={par['q'].std():.4f}  "
        f"range=[{par['q'].min():.4f}, {par['q'].max():.4f}]"
    )
    print("\nper-maturity median q:")
    print(par.groupby("exp")["q"].median().to_string())

    # ---- 4. OTM-only unified ----
    section("[4] OTM-only filter & put→call conversion")
    S0 = float(df["S"].iloc[0])
    is_otm_call = (df.optionType == "call") & (df["K"] > S0)
    is_otm_put = (df.optionType == "put") & (df["K"] < S0)
    otm = df[is_otm_call | is_otm_put].reset_index(drop=True).copy()
    print(f"OTM rows: {len(otm)}  "
          f"(OTM calls: {is_otm_call.sum()}, OTM puts: {is_otm_put.sum()})")
    print(f"dropped (ITM): {len(df) - len(otm)}")

    is_put = otm.optionType == "put"
    otm["call_equiv"] = otm.market_price.astype(float)
    Tp = otm.loc[is_put, "T"].to_numpy()
    Kp = otm.loc[is_put, "K"].to_numpy()
    rp = otm.loc[is_put, "r"].to_numpy()
    Sp = otm.loc[is_put, "S"].to_numpy()
    Pp = otm.loc[is_put, "market_price"].to_numpy()
    otm.loc[is_put, "call_equiv"] = (
        Pp + Sp * np.exp(-q_implied * Tp) - Kp * np.exp(-rp * Tp)
    )

    # consistency check: for matched (K,T) pairs, does converted put match the call?
    section("[4b] Consistency: put-converted vs actual call (matched pairs)")
    consistency = []
    for (exp, K), g in df.groupby(["expirationDate", "K"]):
        c = g[g.optionType == "call"]
        p = g[g.optionType == "put"]
        if len(c) == 1 and len(p) == 1:
            S = float(c["S"].iloc[0])
            T = float(c["T"].iloc[0])
            r = float(c["r"].iloc[0])
            C_actual = float(c.market_price.iloc[0])
            P = float(p.market_price.iloc[0])
            C_converted = P + S * np.exp(-q_implied * T) - K * np.exp(-r * T)
            consistency.append(C_converted - C_actual)
    cons = np.array(consistency)
    print(f"residual (converted - actual call) over {len(cons)} pairs:")
    print(f"  mean={cons.mean():+.4f}  std={cons.std():.4f}  |max|={np.abs(cons).max():.4f}")

    # ---- 5. Implied vol per OTM row ----
    section("[5] Implied volatility (BS with dividend, on call-equivalent)")
    ivs = [
        iv_invert(ce, S, K, T, r, q_implied, "call")
        for ce, S, K, T, r in zip(
            otm.call_equiv, otm["S"], otm["K"], otm["T"], otm["r"]
        )
    ]
    otm["iv"] = ivs
    n_failed = int(otm.iv.isna().sum())
    iv_ok = otm.dropna(subset=["iv"]).reset_index(drop=True)
    print(f"failed: {n_failed} / {len(otm)}")
    print(
        f"IV: range=[{iv_ok.iv.min():.4f}, {iv_ok.iv.max():.4f}]  "
        f"mean={iv_ok.iv.mean():.4f}  std={iv_ok.iv.std():.4f}"
    )
    print("per-maturity IV stats:")
    print(iv_ok.groupby("expirationDate")["iv"].agg(["count", "mean", "std", "min", "max"]).to_string())

    # ---- 6. Save ----
    section("[6] Save")
    iv_ok = iv_ok.assign(q_implied=q_implied)
    out_csv = OUT_DIR / "clean.csv"
    iv_ok.to_csv(out_csv, index=False)
    print(f"saved: {out_csv}  ({len(iv_ok)} rows)")

    # ---- 7. Plot ----
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    cmap = plt.get_cmap("viridis")
    Ts = sorted(iv_ok["T"].unique())
    for i, T in enumerate(Ts):
        sub = iv_ok[iv_ok["T"] == T]
        # use original-source markers for diagnostic
        for ot, marker in [("call", "o"), ("put", "x")]:
            ssub = sub[sub.optionType == ot]
            ax[0].scatter(
                ssub["K"] / S0, ssub.iv,
                color=cmap(i / max(1, len(Ts) - 1)),
                marker=marker, s=30, alpha=0.85,
                label=f"T={T:.3f} {ot}" if i == 0 else None,
            )
    ax[0].set_xlabel("K/S")
    ax[0].set_ylabel("implied vol")
    ax[0].set_title(f"Cleaned OTM IV smile (N={len(iv_ok)})")
    ax[0].legend(fontsize=7, loc="best")

    sc = ax[1].scatter(iv_ok["K"], iv_ok["T"], c=iv_ok.iv, cmap="viridis", s=30)
    plt.colorbar(sc, ax=ax[1], label="IV")
    ax[1].set_xlabel("K")
    ax[1].set_ylabel("T")
    ax[1].set_title("IV surface (cleaned points)")
    plt.tight_layout()
    out_fig = FIG_DIR / "clean_iv.png"
    plt.savefig(out_fig, dpi=600)
    plt.close()
    print(f"saved: {out_fig}")

    section("Summary")
    print(f"raw:                {n0}")
    print(f"after intrinsic:    {n1}  (-{n0 - n1})")
    print(f"after liquidity:    {n2}  (-{n1 - n2})")
    print(f"after OTM-only:     {len(otm)}  (-{n2 - len(otm)})")
    print(f"after IV inversion: {len(iv_ok)}  (-{n_failed})")
    print(f"implied q:          {q_implied:.6f}")


if __name__ == "__main__":
    main()
