"""Numerical estimate of the American early-exercise premium for TSM puts in
our sample, to substantiate the §6 limitation (iv) claim.

For each option we compute the European Black-Scholes price (with dividend) at
its observed implied volatility, then the American price via a Cox-Ross-
Rubinstein binomial tree (1000 steps), and report the difference.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parent.parent
CLEAN = ROOT / "data" / "clean.csv"
OUT = ROOT / "data" / "american_premium.json"

N_STEPS = 1000


def european_bs(S, K, T, r, q, sigma, opt_type):
    if T <= 0 or sigma <= 0:
        return max(0.0, (S - K) if opt_type == "call" else (K - S))
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if opt_type == "call":
        return float(S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
    return float(K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1))


def american_crr(S, K, T, r, q, sigma, opt_type, n_steps=N_STEPS):
    if T <= 0 or sigma <= 0 or n_steps <= 0:
        return max(0.0, (S - K) if opt_type == "call" else (K - S))
    dt = T / n_steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)
    if not (0 < p < 1):
        return float("nan")
    disc = np.exp(-r * dt)
    # terminal asset prices
    j = np.arange(n_steps + 1)
    ST = S * (u ** (n_steps - j)) * (d ** j)
    if opt_type == "call":
        V = np.maximum(ST - K, 0.0)
    else:
        V = np.maximum(K - ST, 0.0)
    # backward induction with early-exercise
    for i in range(n_steps - 1, -1, -1):
        ST = S * (u ** (i - np.arange(i + 1))) * (d ** np.arange(i + 1))
        V = disc * (p * V[:-1] + (1 - p) * V[1:])
        if opt_type == "call":
            intrinsic = np.maximum(ST - K, 0.0)
        else:
            intrinsic = np.maximum(K - ST, 0.0)
        V = np.maximum(V, intrinsic)
    return float(V[0])


def main() -> None:
    df = pd.read_csv(CLEAN)
    print(f"loaded clean: N={len(df)}")
    S = float(df["S"].iloc[0])
    r = float(df["r"].iloc[0])
    q = float(df["q_implied"].iloc[0])

    # We have call_equiv (after parity conversion). For the premium estimate
    # we evaluate both as if the option WERE European (BS with iv) and as
    # American on the *original* option type (call or put as stored in
    # optionType column).
    rows = []
    for _, opt in df.iterrows():
        K, T, iv, ot = float(opt["K"]), float(opt["T"]), float(opt["iv"]), str(opt["optionType"])
        eu = european_bs(S, K, T, r, q, iv, ot)
        am = american_crr(S, K, T, r, q, iv, ot)
        rows.append({"K": K, "T": T, "iv": iv, "type": ot,
                     "european": eu, "american": am, "premium": am - eu})
    out = pd.DataFrame(rows)

    print("\nAmerican − European premium summary:")
    for ot in ["call", "put"]:
        sub = out[out["type"] == ot]
        if len(sub) == 0:
            continue
        print(f"  {ot:4s}: n={len(sub)}  mean={sub.premium.mean():+.4f}  "
              f"std={sub.premium.std():.4f}  max|prem|={sub.premium.abs().max():.4f}  "
              f"frac > $0.10: {(sub.premium.abs() > 0.10).mean():.2%}")

    summary = {
        "S": S, "r": r, "q": q, "n_data": len(out),
        "by_type": {
            ot: {
                "n": int((out["type"] == ot).sum()),
                "mean_premium": float(out.loc[out["type"] == ot, "premium"].mean()),
                "std_premium": float(out.loc[out["type"] == ot, "premium"].std()),
                "max_abs_premium": float(out.loc[out["type"] == ot, "premium"].abs().max()),
                "median_premium": float(out.loc[out["type"] == ot, "premium"].median()),
                "frac_premium_over_10c": float((out.loc[out["type"] == ot, "premium"].abs() > 0.10).mean()),
            }
            for ot in ["call", "put"]
        },
    }
    with open(OUT, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nsaved: {OUT}")


if __name__ == "__main__":
    main()
