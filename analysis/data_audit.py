"""Data audit for features_all.csv.

Reports basic structure, arbitrage violations, put-call parity residuals,
and per-option implied volatility (BS European inversion).
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
FIG_DIR = ROOT / "figures" / "audit"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def bs_price(S: float, K: float, T: float, r: float, sigma: float, opt_type: str) -> float:
    if T <= 0 or sigma <= 0:
        return max(0.0, (S - K) if opt_type == "call" else (K - S))
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if opt_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def implied_vol(price: float, S: float, K: float, T: float, r: float, opt_type: str) -> float:
    intrinsic = max(0.0, (S - K) if opt_type == "call" else (K - S))
    if price < intrinsic - 1e-6:
        return np.nan
    try:
        return brentq(
            lambda s: bs_price(S, K, T, r, s, opt_type) - price,
            1e-6,
            5.0,
            xtol=1e-8,
            maxiter=200,
        )
    except (ValueError, RuntimeError):
        return np.nan


def section(title: str) -> None:
    print()
    print("=" * 64)
    print(title)
    print("=" * 64)


def main() -> None:
    df = pd.read_csv(CSV)

    section("[1] Basic")
    print(f"shape: {df.shape}")
    print(f"NaN total: {df.isna().sum().sum()}")
    print(f"S unique: {df.S.nunique()}  ({df.S.iloc[0]:.4f})")
    print(f"r unique: {df.r.nunique()}  ({df.r.iloc[0]:.4f})")
    print(f"sigma unique (CSV column): {df.sigma.nunique()}  ({df.sigma.iloc[0]:.4f})")
    print(
        df[["K", "T", "market_price", "moneyness"]]
        .describe()
        .T[["min", "max", "mean", "std"]]
        .to_string()
    )

    section("[2] Per-maturity counts")
    rows = []
    for exp, g in df.groupby("expirationDate"):
        rows.append(
            {
                "exp": exp,
                "T": float(g["T"].iloc[0]),
                "n_call": int((g.optionType == "call").sum()),
                "n_put": int((g.optionType == "put").sum()),
                "K_min": float(g.K.min()),
                "K_max": float(g.K.max()),
            }
        )
    print(pd.DataFrame(rows).to_string(index=False))

    section("[3] Below-intrinsic price check (American lower bound)")
    intrinsic = np.where(df.optionType == "call", np.maximum(df.S - df.K, 0), np.maximum(df.K - df.S, 0))
    below = df[df.market_price < intrinsic - 0.01].copy()
    below["intrinsic"] = intrinsic[df.market_price.values < intrinsic - 0.01]
    print(f"below-intrinsic options: {len(below)} / {len(df)}")
    if len(below) > 0:
        print(below[["K", "T", "optionType", "market_price", "intrinsic"]].head(10).to_string(index=False))

    section("[4] Monotonicity violations in K")
    for exp, g in df.groupby("expirationDate"):
        for ot in ["call", "put"]:
            sub = g[g.optionType == ot].sort_values("K")
            if len(sub) < 2:
                continue
            P = sub.market_price.values
            if ot == "call":
                v = int((P[1:] - P[:-1] > 1e-3).sum())
            else:
                v = int((P[:-1] - P[1:] > 1e-3).sum())
            if v > 0:
                print(f"  {exp}  {ot}: {v} violations")
    print("(no output above means no monotonicity violations)")

    section("[5] Butterfly (convexity in K) violations")
    total_butterfly = 0
    for exp, g in df.groupby("expirationDate"):
        for ot in ["call", "put"]:
            sub = g[g.optionType == ot].sort_values("K")
            if len(sub) < 3:
                continue
            K = sub.K.values
            P = sub.market_price.values
            v = 0
            for i in range(1, len(P) - 1):
                w = (K[i + 1] - K[i]) / (K[i + 1] - K[i - 1])
                interp = w * P[i - 1] + (1 - w) * P[i + 1]
                if P[i] > interp + 1e-2:
                    v += 1
            if v > 0:
                print(f"  {exp}  {ot}: {v} butterfly violations")
                total_butterfly += v
    if total_butterfly == 0:
        print("(none)")

    section("[6] Put-call parity (European: C - P = S - K e^{-rT})")
    parities = []
    for (exp, K), g in df.groupby(["expirationDate", "K"]):
        c = g[g.optionType == "call"]
        p = g[g.optionType == "put"]
        if len(c) == 1 and len(p) == 1:
            S = float(c["S"].iloc[0])
            T = float(c["T"].iloc[0])
            r = float(c["r"].iloc[0])
            resid = float(c.market_price.iloc[0] - p.market_price.iloc[0] - (S - K * np.exp(-r * T)))
            parities.append({"exp": exp, "K": K, "T": T, "residual": resid})
    par = pd.DataFrame(parities)
    print(f"matched (K,T) pairs: {len(par)}")
    if len(par) > 0:
        print(
            f"residual: mean={par.residual.mean():+.4f}  "
            f"std={par.residual.std():.4f}  "
            f"|max|={par.residual.abs().max():.4f}"
        )
        print(f"residual quantiles: {par.residual.quantile([0.1, 0.5, 0.9]).to_dict()}")
        print("\nLargest |residual| pairs (top 5):")
        print(par.reindex(par.residual.abs().sort_values(ascending=False).index)
              .head(5).to_string(index=False))

    section("[7] Implied volatility (BS European inversion)")
    df["iv"] = [
        implied_vol(p, S, K, T, r, ot)
        for p, S, K, T, r, ot in zip(
            df.market_price, df.S, df.K, df["T"], df.r, df.optionType
        )
    ]
    n_failed = int(df.iv.isna().sum())
    iv_ok = df.dropna(subset=["iv"])
    print(f"failed inversions: {n_failed} / {len(df)}")
    print(f"IV (all):  range=[{iv_ok.iv.min():.4f}, {iv_ok.iv.max():.4f}]  "
          f"mean={iv_ok.iv.mean():.4f}  std={iv_ok.iv.std():.4f}")
    for ot in ["call", "put"]:
        sub = iv_ok[iv_ok.optionType == ot]
        if len(sub) > 0:
            print(
                f"IV ({ot:4s}): n={len(sub):3d}  "
                f"range=[{sub.iv.min():.4f}, {sub.iv.max():.4f}]  "
                f"mean={sub.iv.mean():.4f}  std={sub.iv.std():.4f}"
            )
    print(f"CSV-provided σ column (single value): {df.sigma.iloc[0]:.4f}")

    section("[8] Call-vs-put IV gap at matched (K, T)")
    gaps = []
    for (exp, K), g in iv_ok.groupby(["expirationDate", "K"]):
        c = g[g.optionType == "call"]
        p = g[g.optionType == "put"]
        if len(c) == 1 and len(p) == 1:
            gaps.append(
                {
                    "exp": exp,
                    "K": K,
                    "T": float(c["T"].iloc[0]),
                    "iv_call": float(c.iv.iloc[0]),
                    "iv_put": float(p.iv.iloc[0]),
                    "gap": float(p.iv.iloc[0] - c.iv.iloc[0]),
                }
            )
    gdf = pd.DataFrame(gaps)
    if len(gdf) > 0:
        print(f"matched (K,T) with both IVs: {len(gdf)}")
        print(
            f"iv_put - iv_call:  mean={gdf.gap.mean():+.4f}  "
            f"std={gdf.gap.std():.4f}  median={gdf.gap.median():+.4f}"
        )
        print("(positive gap is consistent with American early-exercise premium on puts)")

    section("[9] Figures saved to figures/audit/")
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    cmap = plt.get_cmap("viridis")
    Ts = sorted(iv_ok["T"].unique())
    for i, T in enumerate(Ts):
        for ot, marker in [("call", "o"), ("put", "x")]:
            sub = iv_ok[(iv_ok["T"] == T) & (iv_ok.optionType == ot)]
            ax[0].scatter(
                sub.moneyness,
                sub.iv,
                color=cmap(i / max(1, len(Ts) - 1)),
                marker=marker,
                s=28,
                alpha=0.85,
                label=f"T={T:.3f} {ot}" if i == 0 else None,
            )
    ax[0].axhline(df.sigma.iloc[0], color="gray", ls="--", lw=1, label=f"CSV σ={df.sigma.iloc[0]:.3f}")
    ax[0].set_xlabel("moneyness  S/K")
    ax[0].set_ylabel("implied vol")
    ax[0].set_title("IV vs moneyness (color = T; o=call, x=put)")
    ax[0].legend(fontsize=7, loc="best")

    sc = ax[1].scatter(iv_ok.K, iv_ok["T"], c=iv_ok.iv, cmap="viridis", s=30)
    plt.colorbar(sc, ax=ax[1], label="IV")
    ax[1].set_xlabel("K")
    ax[1].set_ylabel("T")
    ax[1].set_title("IV surface points")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "iv_overview.png", dpi=120)
    plt.close()
    print(f"saved: {FIG_DIR / 'iv_overview.png'}")


if __name__ == "__main__":
    main()
