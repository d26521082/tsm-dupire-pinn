"""Quantitative analysis of the PINN-recovered local volatility surface σ_loc(K, T).

Steps:
  1. Retrain the Dupire PINN on the full cleaned dataset (N=192).
  2. Sample σ_loc on a dense (K, T) grid.
  3. Quantify:
       a. ATM term structure σ_loc(K=F_T, T)
       b. Leverage skew strength σ_loc(0.8 S) / σ_loc(S) per maturity
       c. Left-wing power-law fit  σ_loc(K) ~ (K/F)^{-α_L}  for K < F
       d. Right-wing power-law fit σ_loc(K) ~ (K/F)^{+α_R}  for K > F
       e. Local curvature ∂²σ_loc/∂K² distribution
  4. Save metrics + publication-quality figures.
"""
from __future__ import annotations

import json
import sys
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

from train_pinn import (  # noqa: E402
    build_normalizers, make_models, train, C_at, sigma_at,
)

CLEAN_CSV = ROOT / "data" / "clean.csv"
OUT_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures" / "sigma"
FIG_DIR.mkdir(parents=True, exist_ok=True)

PINN_NUM_ITER = 10_000
PINN_N_COLLOC = 256
PINN_SEEDS = [42, 7, 123, 2024, 99]   # ensemble for robustness


def train_one(df, seed: int):
    norm = build_normalizers(df)
    key = jax.random.PRNGKey(seed)
    key_init, key_train = jax.random.split(key)
    model = make_models(key_init)
    model, _ = train(
        model, df, norm, PINN_NUM_ITER, key_train,
        n_colloc=PINN_N_COLLOC, verbose=False,
    )
    return model, norm


def sigma_grid(model, norm, Kg: np.ndarray, Tg: np.ndarray) -> np.ndarray:
    _, sigma_net = model
    KK, TT = np.meshgrid(Kg, Tg)
    K_flat = jnp.asarray(KK.ravel(), dtype=jnp.float32)
    T_flat = jnp.asarray(TT.ravel(), dtype=jnp.float32)
    s = np.asarray(jax.vmap(lambda k, t: sigma_at(sigma_net, k, t, norm))(K_flat, T_flat))
    return s.reshape(KK.shape)


def fit_power_law(K_over_F: np.ndarray, sigma: np.ndarray) -> tuple[float, float, float]:
    """Fit log σ = a - α log(K/F) (so σ ~ (K/F)^{-α}). Returns (alpha, a, R²)."""
    x = np.log(K_over_F)
    y = np.log(sigma)
    res = linregress(x, y)
    return float(-res.slope), float(res.intercept), float(res.rvalue ** 2)


def main() -> None:
    df = pd.read_csv(CLEAN_CSV)
    S = float(df["S"].iloc[0])
    r = float(df["r"].iloc[0])
    q = float(df["q_implied"].iloc[0])
    print(f"loaded clean: N={len(df)}  S={S:.2f}  r={r:.4f}  q={q:.4f}")

    # --------------------------------------------------------------------
    # Train ensemble of PINNs (different seeds) for robustness
    # --------------------------------------------------------------------
    print(f"\ntraining {len(PINN_SEEDS)} PINN ensemble members ...")
    models, norms = [], []
    for s in PINN_SEEDS:
        m, n = train_one(df, s)
        models.append(m)
        norms.append(n)
        print(f"  seed={s} trained")

    # --------------------------------------------------------------------
    # Dense (K, T) grid for surface analysis
    # --------------------------------------------------------------------
    K_min, K_max = float(df["K"].min()), float(df["K"].max())
    T_min, T_max = float(df["T"].min()), float(df["T"].max())
    Kg = np.linspace(K_min, K_max, 120)
    Tg = np.linspace(T_min, T_max, 60)

    # Ensemble mean and std of σ_loc surface
    surfaces = np.stack([sigma_grid(m, n, Kg, Tg) for m, n in zip(models, norms)], axis=0)
    sigma_mean = surfaces.mean(axis=0)
    sigma_std = surfaces.std(axis=0)
    print(f"σ_loc surface  mean range=[{sigma_mean.min():.3f}, {sigma_mean.max():.3f}]  "
          f"ensemble σ range=[{sigma_std.min():.3f}, {sigma_std.max():.3f}]")

    # --------------------------------------------------------------------
    # (a) ATM term structure  σ_loc(K=F_T, T)
    # --------------------------------------------------------------------
    F_T = S * np.exp((r - q) * Tg)
    sigma_atm = np.array([
        np.interp(F_T[j], Kg, sigma_mean[j]) for j in range(len(Tg))
    ])
    sigma_atm_std = np.array([
        np.interp(F_T[j], Kg, sigma_std[j]) for j in range(len(Tg))
    ])
    print(f"\nATM term structure σ_loc(F, T):")
    print(f"  T=[{Tg[0]:.3f}, {Tg[-1]:.3f}]  σ=[{sigma_atm.min():.3f}, {sigma_atm.max():.3f}]")

    # --------------------------------------------------------------------
    # (b) Leverage skew strength  σ_loc(0.8 F) / σ_loc(F)
    # --------------------------------------------------------------------
    skew = np.array([
        np.interp(0.8 * F_T[j], Kg, sigma_mean[j]) /
        np.interp(F_T[j], Kg, sigma_mean[j])
        for j in range(len(Tg))
    ])
    print(f"\nLeverage skew σ_loc(0.8F)/σ_loc(F):")
    print(f"  range=[{skew.min():.3f}, {skew.max():.3f}]  median={np.median(skew):.3f}")

    # --------------------------------------------------------------------
    # (c-d) Power-law wing exponents
    # --------------------------------------------------------------------
    Ts_data = sorted(df["T"].unique())
    wing_fits = []
    for T_target in Ts_data:
        # use the grid row closest to T_target
        j = int(np.argmin(np.abs(Tg - T_target)))
        F = S * np.exp((r - q) * Tg[j])
        sigma_row = sigma_mean[j]
        kf = Kg / F
        # Left wing: K/F in [0.5, 0.95]
        left_mask = (kf > 0.4) & (kf < 0.95)
        if left_mask.sum() >= 3:
            alpha_L, a_L, R2_L = fit_power_law(kf[left_mask], sigma_row[left_mask])
        else:
            alpha_L = a_L = R2_L = np.nan
        # Right wing: K/F in [1.05, 1.5]
        right_mask = (kf > 1.05) & (kf < 1.6)
        if right_mask.sum() >= 3:
            # σ ~ (K/F)^{+α_R} → fit log σ = a + α_R log(K/F)
            x = np.log(kf[right_mask])
            y = np.log(sigma_row[right_mask])
            res = linregress(x, y)
            alpha_R, a_R, R2_R = float(res.slope), float(res.intercept), float(res.rvalue ** 2)
        else:
            alpha_R = a_R = R2_R = np.nan
        wing_fits.append({
            "T": float(Tg[j]),
            "alpha_L": alpha_L, "R2_L": R2_L,
            "alpha_R": alpha_R, "R2_R": R2_R,
        })

    # --------------------------------------------------------------------
    # (e) Curvature distribution  ∂²σ/∂K²
    # --------------------------------------------------------------------
    dK = Kg[1] - Kg[0]
    d2_sigma_dK2 = (
        sigma_mean[:, 2:] - 2 * sigma_mean[:, 1:-1] + sigma_mean[:, :-2]
    ) / dK**2
    curv_max_per_T = d2_sigma_dK2.max(axis=1)
    K_at_curv_max = Kg[1:-1][np.argmax(d2_sigma_dK2, axis=1)]
    print(f"\nCurvature ∂²σ/∂K²:")
    print(f"  max range=[{curv_max_per_T.min():.4e}, {curv_max_per_T.max():.4e}]")
    print(f"  K at max curvature (per T): [{K_at_curv_max.min():.0f}, {K_at_curv_max.max():.0f}]")

    # --------------------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------------------
    summary = {
        "n_data": len(df),
        "n_seeds": len(PINN_SEEDS),
        "S": S, "r": r, "q": q,
        "K_range": [K_min, K_max],
        "T_range": [T_min, T_max],
        "sigma_loc_overall": {
            "mean_min": float(sigma_mean.min()),
            "mean_max": float(sigma_mean.max()),
            "ensemble_std_min": float(sigma_std.min()),
            "ensemble_std_max": float(sigma_std.max()),
        },
        "atm_term_structure": {
            "T_grid": Tg.tolist(),
            "sigma_loc_atm": sigma_atm.tolist(),
            "sigma_loc_atm_ensemble_std": sigma_atm_std.tolist(),
        },
        "leverage_skew": {
            "T_grid": Tg.tolist(),
            "skew_0p8_over_atm": skew.tolist(),
            "median": float(np.median(skew)),
        },
        "wing_exponents": wing_fits,
        "curvature": {
            "K_at_max_per_T": K_at_curv_max.tolist(),
            "T_grid_for_K_max": Tg.tolist(),
            "max_curv_per_T": curv_max_per_T.tolist(),
        },
    }
    with open(OUT_DIR / "sigma_analysis.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nsaved: {OUT_DIR / 'sigma_analysis.json'}")

    # --------------------------------------------------------------------
    # Plots
    # --------------------------------------------------------------------
    # Surface (mean) with ensemble std as transparency overlay
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    cs = ax[0].contourf(Kg, Tg, sigma_mean, levels=20, cmap="plasma")
    ax[0].plot(F_T, Tg, "k--", lw=1, label="forward F_T")
    plt.colorbar(cs, ax=ax[0], label="σ_loc")
    ax[0].set_xlabel("K"); ax[0].set_ylabel("T")
    ax[0].set_title(f"σ_loc(K, T)  (ensemble mean over {len(PINN_SEEDS)} seeds)")
    ax[0].legend(loc="upper right", fontsize=8)

    cs2 = ax[1].contourf(Kg, Tg, sigma_std, levels=20, cmap="viridis")
    plt.colorbar(cs2, ax=ax[1], label="σ_loc ensemble std")
    ax[1].set_xlabel("K"); ax[1].set_ylabel("T")
    ax[1].set_title("Ensemble uncertainty")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "surface_ensemble.png", dpi=140)
    plt.close()

    # σ_loc smiles per maturity
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.get_cmap("viridis")
    for i, T_target in enumerate(Ts_data):
        j = int(np.argmin(np.abs(Tg - T_target)))
        F = F_T[j]
        kf = Kg / F
        ax.plot(kf, sigma_mean[j], color=cmap(i / max(1, len(Ts_data) - 1)),
                lw=1.5, label=f"T={Tg[j]:.3f}")
        ax.fill_between(kf, sigma_mean[j] - sigma_std[j], sigma_mean[j] + sigma_std[j],
                        color=cmap(i / max(1, len(Ts_data) - 1)), alpha=0.15)
    ax.axvline(1.0, color="k", ls="--", lw=0.8, alpha=0.5)
    ax.set_xlabel("K/F"); ax.set_ylabel("σ_loc")
    ax.set_title("Local-volatility smile per maturity (with ensemble band)")
    ax.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "smile_per_T.png", dpi=140)
    plt.close()

    # ATM term structure
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(Tg, sigma_atm, "k-", lw=2, label="σ_loc(F_T, T)")
    ax.fill_between(Tg, sigma_atm - sigma_atm_std, sigma_atm + sigma_atm_std,
                    color="gray", alpha=0.3, label="ensemble ±1σ")
    ax.set_xlabel("T (years)"); ax.set_ylabel("σ_loc(F_T, T)")
    ax.set_title("ATM local-vol term structure")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "atm_term_structure.png", dpi=140)
    plt.close()

    # Wing exponents per T
    fig, ax = plt.subplots(figsize=(8, 5))
    Ts_arr = np.array([w["T"] for w in wing_fits])
    aL = np.array([w["alpha_L"] for w in wing_fits])
    aR = np.array([w["alpha_R"] for w in wing_fits])
    R2L = np.array([w["R2_L"] for w in wing_fits])
    R2R = np.array([w["R2_R"] for w in wing_fits])
    ax.plot(Ts_arr, aL, "ro-", label=r"$\alpha_L$ (left wing)")
    ax.plot(Ts_arr, aR, "bo-", label=r"$\alpha_R$ (right wing)")
    ax.set_xlabel("T (years)"); ax.set_ylabel("power-law exponent")
    ax.set_title("Power-law wing exponents  σ_loc(K) ~ (K/F)^{∓α}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    # secondary axis for R²
    ax2 = ax.twinx()
    ax2.plot(Ts_arr, R2L, "r:", alpha=0.6, label="R² L")
    ax2.plot(Ts_arr, R2R, "b:", alpha=0.6, label="R² R")
    ax2.set_ylabel("R²")
    ax2.set_ylim(0, 1.05)
    ax2.legend(loc="lower right", fontsize=7)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "wing_exponents.png", dpi=140)
    plt.close()

    # Skew strength
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(Tg, skew, "k-", lw=2)
    ax.axhline(1.0, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("T (years)")
    ax.set_ylabel(r"$\sigma_{loc}(0.8 F)\,/\,\sigma_{loc}(F)$")
    ax.set_title("Leverage skew strength vs maturity")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "skew_term_structure.png", dpi=140)
    plt.close()

    print(f"saved: {FIG_DIR}/*.png")
    print("\n=== Summary ===")
    print(f"  σ_loc range: [{sigma_mean.min():.3f}, {sigma_mean.max():.3f}]  "
          f"ensemble std max: {sigma_std.max():.3f}")
    print(f"  ATM σ_loc(F, T): [{sigma_atm.min():.3f}, {sigma_atm.max():.3f}]")
    print(f"  Leverage skew (median across T): {np.median(skew):.3f}")
    aL_clean = aL[~np.isnan(aL)]
    aR_clean = aR[~np.isnan(aR)]
    if len(aL_clean):
        print(f"  Left-wing α_L:  range=[{aL_clean.min():.3f}, {aL_clean.max():.3f}]  "
              f"median={np.median(aL_clean):.3f}  median R²={np.median(R2L[~np.isnan(R2L)]):.3f}")
    if len(aR_clean):
        print(f"  Right-wing α_R: range=[{aR_clean.min():.3f}, {aR_clean.max():.3f}]  "
              f"median={np.median(aR_clean):.3f}  median R²={np.median(R2R[~np.isnan(R2R)]):.3f}")


if __name__ == "__main__":
    main()
