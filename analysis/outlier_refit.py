"""Sanity refit excluding the two IV-outliers identified in §2.5.

Trains a single PINN on clean.csv minus the two extreme deep-OTM-put observations
(K=105 T=0.066, K=85 T=0.142) and reports whether the calibration is
materially affected.
"""
from __future__ import annotations

import sys
from pathlib import Path

import jax
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))

from train_pinn import build_normalizers, make_models, train  # noqa: E402
from cv_evaluate import predict_pinn  # noqa: E402

SEED = 42
NUM_ITER = 10_000
N_COLLOC = 256


def main() -> None:
    df = pd.read_csv(ROOT / "data" / "clean.csv")
    print(f"loaded clean: N={len(df)}")

    # Identify the two outliers (IV > 1.0, deep-OTM puts)
    outlier_mask = (
        ((df["K"] == 105.0) & (np.isclose(df["T"], 0.065753, atol=1e-4))) |
        ((df["K"] == 85.0) & (np.isclose(df["T"], 0.142466, atol=1e-4)))
    )
    print(f"outliers identified: {int(outlier_mask.sum())}")
    print(df[outlier_mask][["K", "T", "optionType", "iv", "call_equiv"]].to_string(index=False))

    df_minus = df[~outlier_mask].reset_index(drop=True)
    print(f"\n--- training on N = {len(df)} (full) ---")
    norm = build_normalizers(df)
    key = jax.random.PRNGKey(SEED)
    ki, kt = jax.random.split(key)
    model_full = make_models(ki)
    model_full, _ = train(model_full, df, norm, NUM_ITER, kt,
                          n_colloc=N_COLLOC, verbose=False)
    pred_full = predict_pinn(model_full, df["K"].to_numpy(), df["T"].to_numpy(), norm)
    err_full = pred_full - df["call_equiv"].to_numpy()
    rmse_full = float(np.sqrt(np.mean(err_full ** 2)))
    print(f"in-sample RMSE = {rmse_full:.4f}")

    print(f"\n--- training on N = {len(df_minus)} (outliers removed) ---")
    norm_m = build_normalizers(df_minus)
    key = jax.random.PRNGKey(SEED)
    ki, kt = jax.random.split(key)
    model_m = make_models(ki)
    model_m, _ = train(model_m, df_minus, norm_m, NUM_ITER, kt,
                       n_colloc=N_COLLOC, verbose=False)
    pred_m = predict_pinn(model_m, df_minus["K"].to_numpy(), df_minus["T"].to_numpy(), norm_m)
    err_m = pred_m - df_minus["call_equiv"].to_numpy()
    rmse_m = float(np.sqrt(np.mean(err_m ** 2)))
    print(f"in-sample RMSE (on N=190) = {rmse_m:.4f}")

    # Predictions of model_m at the SAME data points as the full model — predict
    # at the 2 held-out points too to see how the model would treat them
    pred_m_full_grid = predict_pinn(model_m, df["K"].to_numpy(), df["T"].to_numpy(), norm_m)
    err_m_full = pred_m_full_grid - df["call_equiv"].to_numpy()
    rmse_m_full = float(np.sqrt(np.mean(err_m_full ** 2)))
    rmse_m_at_outliers = float(np.sqrt(np.mean(err_m_full[outlier_mask] ** 2)))
    rmse_full_at_outliers = float(np.sqrt(np.mean(err_full[outlier_mask] ** 2)))
    print(f"\n--- comparison ---")
    print(f"full PINN  RMSE (192 pts) = {rmse_full:.4f}")
    print(f"refit PINN evaluated on 192 pts = {rmse_m_full:.4f}")
    print(f"  ... at the 2 outliers only: full={rmse_full_at_outliers:.4f}  refit={rmse_m_at_outliers:.4f}")
    print(f"  ... at 190 non-outlier pts:  full={float(np.sqrt(np.mean(err_full[~outlier_mask] ** 2))):.4f}  refit={float(np.sqrt(np.mean(err_m_full[~outlier_mask] ** 2))):.4f}")


if __name__ == "__main__":
    main()
