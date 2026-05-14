"""PDE-prior ablation: isolate which loss components carry out-of-sample weight.

Trains four PINN variants and runs LOMO cross-validation for each:
    A. data only           (w_pde = 0,    w_arb = 0)
    B. data + arbitrage    (w_pde = 0,    w_arb = full)
    C. data + PDE          (w_pde = full, w_arb = 0)
    D. data + PDE + arb    (full = the paper's PINN)

Reports in-sample and OOS RMSE / MAE per variant.
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

CLEAN_CSV = ROOT / "data" / "clean.csv"
OUT_DIR = ROOT / "data"

NUM_ITER = 5_000
N_COLLOC = 256
SEED = 42

VARIANTS = {
    "A_data_only":  dict(w_pde_max=0.0,  w_arb_max=0.0),
    "B_data_arb":   dict(w_pde_max=0.0,  w_arb_max=0.05),
    "C_data_pde":   dict(w_pde_max=0.01, w_arb_max=0.0),
    "D_full":       dict(w_pde_max=0.01, w_arb_max=0.05),
}


def lomo_one_variant(df: pd.DataFrame, variant_name: str, kwargs: dict) -> dict:
    Ts = sorted(df["T"].unique())
    rows = []
    for fold_i, held_T in enumerate(Ts):
        train_df = df[df["T"] != held_T].reset_index(drop=True)
        test_df = df[df["T"] == held_T].reset_index(drop=True)
        norm = build_normalizers(train_df)
        key = jax.random.PRNGKey(SEED + fold_i)
        ki, kt = jax.random.split(key)
        model = make_models(ki)
        model, _ = train(
            model, train_df, norm, NUM_ITER, kt,
            n_colloc=N_COLLOC, verbose=False, print_every=NUM_ITER + 1, **kwargs,
        )
        c_pred = predict_pinn(model, test_df["K"].to_numpy(), test_df["T"].to_numpy(), norm)
        c_obs = test_df["call_equiv"].to_numpy()
        for i in range(len(test_df)):
            rows.append({
                "variant": variant_name, "fold": fold_i, "held_T": held_T,
                "K": float(test_df["K"].iloc[i]), "T": float(test_df["T"].iloc[i]),
                "C_obs": float(c_obs[i]), "C_pred": float(c_pred[i]),
                "resid": float(c_pred[i] - c_obs[i]),
            })
    return rows


def in_sample_one_variant(df: pd.DataFrame, kwargs: dict) -> dict:
    norm = build_normalizers(df)
    key = jax.random.PRNGKey(SEED)
    ki, kt = jax.random.split(key)
    model = make_models(ki)
    print("    [in-sample] training ...", flush=True)
    model, _ = train(model, df, norm, NUM_ITER, kt, n_colloc=N_COLLOC,
                     verbose=True, print_every=1000, **kwargs)
    c_pred = predict_pinn(model, df["K"].to_numpy(), df["T"].to_numpy(), norm)
    err = c_pred - df["call_equiv"].to_numpy()
    return {
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "max_abs": float(np.max(np.abs(err))),
    }


def main() -> None:
    df = pd.read_csv(CLEAN_CSV)
    print(f"loaded clean: N={len(df)}")
    print(f"variants: {list(VARIANTS.keys())}")

    summary = {}
    all_lomo = []
    for name, kwargs in VARIANTS.items():
        print(f"\n=== {name}  ({kwargs}) ===")
        t0 = time.time()
        in_s = in_sample_one_variant(df, kwargs)
        print(f"  in-sample RMSE={in_s['rmse']:.4f}  MAE={in_s['mae']:.4f}  ({time.time()-t0:.0f}s)")
        t0 = time.time()
        lomo_rows = lomo_one_variant(df, name, kwargs)
        all_lomo.extend(lomo_rows)
        resid = np.array([r["resid"] for r in lomo_rows])
        oos_rmse = float(np.sqrt(np.mean(resid ** 2)))
        oos_mae = float(np.mean(np.abs(resid)))
        print(f"  LOMO OOS RMSE={oos_rmse:.4f}  MAE={oos_mae:.4f}  ({time.time()-t0:.0f}s)")
        summary[name] = {
            "in_sample": in_s,
            "oos": {"rmse": oos_rmse, "mae": oos_mae, "n": len(resid)},
            "config": kwargs,
        }

    # Save
    with open(OUT_DIR / "ablation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    pd.DataFrame(all_lomo).to_csv(OUT_DIR / "ablation_lomo_predictions.csv", index=False)
    print(f"\nsaved: {OUT_DIR / 'ablation_summary.json'}")
    print(f"saved: {OUT_DIR / 'ablation_lomo_predictions.csv'}")

    print("\n=== Ablation summary ===")
    print(f"{'variant':16s}  {'in-sample RMSE':>14s}  {'OOS RMSE':>10s}  {'OOS MAE':>10s}")
    for name, info in summary.items():
        print(f"{name:16s}  {info['in_sample']['rmse']:>14.4f}  "
              f"{info['oos']['rmse']:>10.4f}  {info['oos']['mae']:>10.4f}")


if __name__ == "__main__":
    main()
