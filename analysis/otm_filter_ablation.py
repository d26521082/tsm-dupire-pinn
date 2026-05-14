"""Ablation of the OTM-only filter: train the PINN with and without filtering
to ITM options excluded, compare in-sample and LOMO out-of-sample performance.

The full-strike variant uses both calls and puts directly (no parity unification),
which means the model sees both call and put prices at every observed (K, T)
without conversion. To keep the call surface consistent, we use only call options
across all strikes (no parity-converted puts) — this is the natural "no OTM-only
filter" comparison.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import jax
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))

from train_pinn import build_normalizers, make_models, train  # noqa: E402
from cv_evaluate import predict_pinn  # noqa: E402

RAW_CSV = ROOT / "features_all.csv"
CLEAN = ROOT / "data" / "clean.csv"
OUT = ROOT / "data" / "otm_ablation.json"

SEED = 42
NUM_ITER = 10_000
N_COLLOC = 256


def all_strike_calls(raw_df: pd.DataFrame, q: float) -> pd.DataFrame:
    """Return ALL call options (both ITM and OTM) from the raw CSV, with the
    sub-cent-liquidity and below-intrinsic filters applied as in data_prep."""
    df = raw_df.copy()
    intrinsic = np.where(df.optionType == "call",
                         np.maximum(df.S - df.K, 0),
                         np.maximum(df.K - df.S, 0))
    df = df[df.market_price >= intrinsic - 0.01]
    df = df[df.market_price >= 0.05]
    df = df[df.optionType == "call"].reset_index(drop=True)
    df["call_equiv"] = df["market_price"]
    df["q_implied"] = q
    # Need iv column for compatibility (placeholder; not used in training)
    df["iv"] = 0.3
    return df


def main() -> None:
    raw = pd.read_csv(RAW_CSV)
    clean = pd.read_csv(CLEAN)
    q = float(clean["q_implied"].iloc[0])
    all_calls = all_strike_calls(raw, q)
    print(f"clean (OTM-only unification): N={len(clean)}")
    print(f"all-strike calls (no OTM filter): N={len(all_calls)}")

    # in-sample: train on each, evaluate on each's own data
    summary = {}
    for name, df_train in [("OTM_only_unified", clean), ("all_strike_calls", all_calls)]:
        t0 = time.time()
        norm = build_normalizers(df_train)
        key = jax.random.PRNGKey(SEED)
        ki, kt = jax.random.split(key)
        model = make_models(ki)
        model, _ = train(model, df_train, norm, NUM_ITER, kt, n_colloc=N_COLLOC, verbose=False)
        c_pred = predict_pinn(model, df_train["K"].to_numpy(), df_train["T"].to_numpy(), norm)
        err = c_pred - df_train["call_equiv"].to_numpy()
        rmse = float(np.sqrt(np.mean(err ** 2)))
        mae = float(np.mean(np.abs(err)))
        summary[name] = {"n": len(df_train), "in_sample_rmse": rmse, "in_sample_mae": mae,
                         "train_time_s": time.time() - t0}
        print(f"  {name}: in-sample RMSE={rmse:.4f}  MAE={mae:.4f}  ({time.time()-t0:.0f}s)")

    with open(OUT, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nsaved: {OUT}")


if __name__ == "__main__":
    main()
