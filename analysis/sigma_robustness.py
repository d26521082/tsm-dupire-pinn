"""Robustness checks for the recovered local-volatility wing exponents:
  (1) T-adaptive K window — restricts each maturity's wing fit to its actually
      observed K range, eliminating extrapolation contamination.
  (2) Window sensitivity — re-fits with three K/F windows.
  (3) Data-bootstrap uncertainty — resamples observations with replacement,
      retrains PINN, recomputes alpha_L per maturity.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))

from train_pinn import build_normalizers, make_models, train, sigma_at  # noqa: E402

CLEAN_CSV = ROOT / "data" / "clean.csv"
OUT_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures" / "robustness"
FIG_DIR.mkdir(parents=True, exist_ok=True)

NUM_ITER = 10_000
N_COLLOC = 256
N_BOOT = 10                       # data-bootstrap samples
WINDOWS = [(0.3, 0.95), (0.4, 0.95), (0.5, 0.9)]
SEED = 42


def fit_wing_left(K_over_F: np.ndarray, sigma: np.ndarray, lo: float, hi: float) -> tuple[float, float, int]:
    """Returns (alpha_L, R^2, n_used)."""
    mask = (K_over_F > lo) & (K_over_F < hi)
    if mask.sum() < 3:
        return float("nan"), float("nan"), int(mask.sum())
    x = np.log(K_over_F[mask])
    y = np.log(sigma[mask])
    res = linregress(x, y)
    return float(-res.slope), float(res.rvalue ** 2), int(mask.sum())


def alpha_L_per_T(model, norm, df, n_K_grid: int = 200,
                  window: tuple[float, float] = (0.4, 0.95),
                  T_adaptive: bool = False) -> list[dict]:
    """For each maturity, sample sigma_loc on a K grid. If T_adaptive, the grid
    is restricted to the observed K range AT THAT MATURITY; otherwise to the
    overall K range. Wing fit on K/F within (window[0], window[1])."""
    _, sigma_net = model
    S, r, q = norm["S"], norm["r"], norm["q"]
    fits = []
    for T in sorted(df["T"].unique()):
        F = S * np.exp((r - q) * T)
        if T_adaptive:
            obs_K = df[df["T"] == T]["K"].to_numpy()
            K_lo, K_hi = float(obs_K.min()), float(obs_K.max())
        else:
            K_lo, K_hi = float(df["K"].min()), float(df["K"].max())
        Kg = np.linspace(K_lo, K_hi, n_K_grid)
        K_flat = jnp.asarray(Kg, dtype=jnp.float32)
        T_flat = jnp.full_like(K_flat, T, dtype=jnp.float32)
        s = np.asarray(jax.vmap(lambda k, t: sigma_at(sigma_net, k, t, norm))(K_flat, T_flat))
        kf = Kg / F
        a_L, R2, n_pts = fit_wing_left(kf, s, *window)
        fits.append({
            "T": float(T),
            "K_range_used": [K_lo, K_hi],
            "alpha_L": a_L,
            "R2_L": R2,
            "n_fit_points": n_pts,
            "T_adaptive": T_adaptive,
            "window": list(window),
        })
    return fits


def train_one_pinn(df: pd.DataFrame, seed: int):
    norm = build_normalizers(df)
    key = jax.random.PRNGKey(seed)
    ki, kt = jax.random.split(key)
    model = make_models(ki)
    model, _ = train(model, df, norm, NUM_ITER, kt, n_colloc=N_COLLOC, verbose=False)
    return model, norm


def main() -> None:
    df = pd.read_csv(CLEAN_CSV)
    print(f"loaded clean: N={len(df)}")

    # ---- baseline run (original window, original grid) ----
    print("\n[1] baseline single-seed PINN (T-adaptive + non-adaptive comparison)")
    t0 = time.time()
    model, norm = train_one_pinn(df, SEED)
    print(f"  trained ({time.time()-t0:.0f}s)")

    fits_nonadapt = alpha_L_per_T(model, norm, df, T_adaptive=False, window=(0.4, 0.95))
    fits_adapt = alpha_L_per_T(model, norm, df, T_adaptive=True, window=(0.4, 0.95))

    print(f"  α_L non-adaptive: {[round(f['alpha_L'], 3) for f in fits_nonadapt]}")
    print(f"  α_L T-adaptive  : {[round(f['alpha_L'], 3) for f in fits_adapt]}")

    # ---- window sensitivity (single seed, T-adaptive) ----
    print("\n[2] window sensitivity (single seed, T-adaptive)")
    fits_by_window = {}
    for w in WINDOWS:
        fits_by_window[f"{w[0]:.2f}_{w[1]:.2f}"] = alpha_L_per_T(
            model, norm, df, T_adaptive=True, window=w)
        print(f"  window={w}: α_L median={np.nanmedian([f['alpha_L'] for f in fits_by_window[f'{w[0]:.2f}_{w[1]:.2f}']]):.3f}")

    # ---- data bootstrap uncertainty (single window, T-adaptive) ----
    print(f"\n[3] data-bootstrap uncertainty: {N_BOOT} resamples")
    rng = np.random.default_rng(SEED)
    n = len(df)
    boot_alphas = []   # shape (N_BOOT, n_T)
    Ts = sorted(df["T"].unique())
    for b in range(N_BOOT):
        # bootstrap-with-replacement preserving total size
        idx = rng.integers(0, n, size=n)
        boot_df = df.iloc[idx].reset_index(drop=True)
        # ensure all maturities present (if any missing, resample again)
        missing = set(Ts) - set(boot_df["T"].unique())
        if missing:
            for T in missing:
                # add one observation from that maturity to avoid degeneracy
                boot_df = pd.concat([boot_df, df[df["T"] == T].sample(1, random_state=b)],
                                    ignore_index=True)
        t0 = time.time()
        m_b, n_b = train_one_pinn(boot_df, SEED + 1000 + b)
        fits_b = alpha_L_per_T(m_b, n_b, boot_df, T_adaptive=True, window=(0.4, 0.95))
        boot_alphas.append([f["alpha_L"] for f in fits_b])
        print(f"  bootstrap {b+1}/{N_BOOT}  α_L={[round(a, 2) for a in boot_alphas[-1]]}  ({time.time()-t0:.0f}s)")
    boot_alphas = np.array(boot_alphas)
    boot_mean = np.nanmean(boot_alphas, axis=0)
    boot_std = np.nanstd(boot_alphas, axis=0)
    print(f"\n  data-bootstrap α_L mean: {[round(a, 3) for a in boot_mean]}")
    print(f"  data-bootstrap α_L std : {[round(a, 3) for a in boot_std]}")

    # ---- save ----
    summary = {
        "n_data": len(df),
        "n_bootstrap": N_BOOT,
        "n_iter_per_train": NUM_ITER,
        "window_default": [0.4, 0.95],
        "non_adaptive_fits": fits_nonadapt,
        "T_adaptive_fits": fits_adapt,
        "window_sensitivity": fits_by_window,
        "data_bootstrap": {
            "T_grid": Ts,
            "alpha_L_mean": boot_mean.tolist(),
            "alpha_L_std": boot_std.tolist(),
            "alpha_L_all_runs": boot_alphas.tolist(),
        },
    }
    with open(OUT_DIR / "sigma_robustness.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nsaved: {OUT_DIR / 'sigma_robustness.json'}")

    # ---- plots ----
    # alpha_L vs T comparison (non-adaptive vs T-adaptive)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot([f["T"] for f in fits_nonadapt], [f["alpha_L"] for f in fits_nonadapt],
            "ro-", label="non-adaptive K window", lw=1.5)
    ax.plot([f["T"] for f in fits_adapt], [f["alpha_L"] for f in fits_adapt],
            "bs-", label="T-adaptive K window", lw=1.5)
    ax.errorbar(Ts, boot_mean, yerr=boot_std, fmt="ko", capsize=4,
                label=f"data-bootstrap mean ± std (n={N_BOOT})")
    ax.set_xlabel("T (years)")
    ax.set_ylabel(r"$\alpha_L$ (left-wing power-law exponent)")
    ax.set_title(r"$\alpha_L(T)$ — non-adaptive vs T-adaptive K window, with data-bootstrap uncertainty")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "alpha_L_T_adaptive_vs_bootstrap.png", dpi=600)
    plt.close()

    # window sensitivity overlay
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for w_key, fits in fits_by_window.items():
        ax.plot([f["T"] for f in fits], [f["alpha_L"] for f in fits],
                "o-", label=f"K/F ∈ ({w_key.replace('_', ', ')})", lw=1.5)
    ax.set_xlabel("T (years)")
    ax.set_ylabel(r"$\alpha_L$")
    ax.set_title("Sensitivity of $\\alpha_L(T)$ to fit-window choice (T-adaptive grid)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "alpha_L_window_sensitivity.png", dpi=600)
    plt.close()
    print(f"saved: {FIG_DIR}/*.png")


if __name__ == "__main__":
    main()
