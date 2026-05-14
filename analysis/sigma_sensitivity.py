"""Sensitivity of recovered local-vol structure to the BS-baseline σ_ref.

For each σ_ref ∈ {0.30, 0.35, 0.40}, train a 3-seed ensemble of PINNs and
extract the per-maturity left-wing power-law exponent α_L. Plot α_L(T) as
three overlaid curves to demonstrate (in)sensitivity of the headline finding
to the baseline-vol choice.
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

import train_pinn  # noqa: E402
from train_pinn import (  # noqa: E402
    build_normalizers, make_models, train, sigma_at,
)

CLEAN_CSV = ROOT / "data" / "clean.csv"
OUT_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures" / "sensitivity"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SIGMA_REF_VALUES = [0.30, 0.35, 0.40]
SEEDS = [42, 7, 2024]
NUM_ITER = 10_000
N_COLLOC = 256


def fit_left_wing(K_over_F: np.ndarray, sigma: np.ndarray) -> tuple[float, float]:
    """Fit log σ = a - α_L log(K/F). Returns (alpha_L, R²)."""
    mask = (K_over_F > 0.4) & (K_over_F < 0.95)
    if mask.sum() < 3:
        return float("nan"), float("nan")
    x = np.log(K_over_F[mask])
    y = np.log(sigma[mask])
    res = linregress(x, y)
    return float(-res.slope), float(res.rvalue ** 2)


def alpha_L_per_T(model, norm, df, Kg, Tg) -> list[dict]:
    _, sigma_net = model
    KK, TT = np.meshgrid(Kg, Tg)
    K_flat = jnp.asarray(KK.ravel(), dtype=jnp.float32)
    T_flat = jnp.asarray(TT.ravel(), dtype=jnp.float32)
    s = np.asarray(jax.vmap(lambda k, t: sigma_at(sigma_net, k, t, norm))(K_flat, T_flat))
    s = s.reshape(KK.shape)
    S, r, q = norm["S"], norm["r"], norm["q"]

    fits = []
    for T_target in sorted(df["T"].unique()):
        j = int(np.argmin(np.abs(Tg - T_target)))
        F = S * np.exp((r - q) * Tg[j])
        kf = Kg / F
        a, r2 = fit_left_wing(kf, s[j])
        fits.append({"T": float(Tg[j]), "alpha_L": a, "R2_L": r2})
    return fits


def main() -> None:
    df = pd.read_csv(CLEAN_CSV)
    print(f"loaded clean: N={len(df)}")

    K_min, K_max = float(df["K"].min()), float(df["K"].max())
    T_min, T_max = float(df["T"].min()), float(df["T"].max())
    Kg = np.linspace(K_min, K_max, 120)
    Tg = np.linspace(T_min, T_max, 60)

    results: dict[str, list[list[dict]]] = {f"{sr:.2f}": [] for sr in SIGMA_REF_VALUES}

    for sr in SIGMA_REF_VALUES:
        train_pinn.SIGMA_REF = sr   # monkey-patch the module global
        print(f"\n=== σ_ref = {sr:.2f} ===")
        for seed in SEEDS:
            norm = build_normalizers(df)
            key = jax.random.PRNGKey(seed)
            ki, kt = jax.random.split(key)
            model = make_models(ki)
            model, _ = train(
                model, df, norm, NUM_ITER, kt,
                n_colloc=N_COLLOC, verbose=False,
            )
            fits = alpha_L_per_T(model, norm, df, Kg, Tg)
            results[f"{sr:.2f}"].append(fits)
            aL_arr = np.array([f["alpha_L"] for f in fits])
            print(f"  seed={seed}  α_L median={np.nanmedian(aL_arr):.3f}  range=[{np.nanmin(aL_arr):.3f}, {np.nanmax(aL_arr):.3f}]")

    # ---- aggregate (mean over seeds, per σ_ref, per T) ----
    summary = {}
    for sr_key, seed_runs in results.items():
        Ts = [f["T"] for f in seed_runs[0]]
        aL_per_seed = np.array([[f["alpha_L"] for f in run] for run in seed_runs])  # (n_seed, n_T)
        aL_mean = np.nanmean(aL_per_seed, axis=0)
        aL_std = np.nanstd(aL_per_seed, axis=0)
        summary[sr_key] = {
            "T": Ts,
            "alpha_L_mean": aL_mean.tolist(),
            "alpha_L_std": aL_std.tolist(),
            "alpha_L_median_overall": float(np.nanmedian(aL_per_seed)),
            "alpha_L_min_overall": float(np.nanmin(aL_per_seed)),
            "alpha_L_max_overall": float(np.nanmax(aL_per_seed)),
        }

    with open(OUT_DIR / "sigma_sensitivity.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nsaved: {OUT_DIR / 'sigma_sensitivity.json'}")

    # ---- plot α_L(T) overlay ----
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    colors = ["tab:blue", "tab:orange", "tab:green"]
    for color, (sr_key, info) in zip(colors, summary.items()):
        T = np.array(info["T"])
        m = np.array(info["alpha_L_mean"])
        s = np.array(info["alpha_L_std"])
        ax.plot(T, m, "o-", color=color, lw=2, label=rf"$\sigma_{{\mathrm{{ref}}}} = {sr_key}$")
        ax.fill_between(T, m - s, m + s, color=color, alpha=0.18)
    ax.set_xlabel("T (years)")
    ax.set_ylabel(r"$\alpha_L$ (left-wing power-law exponent)")
    ax.set_title(r"Sensitivity of $\alpha_L(T)$ to BS baseline volatility $\sigma_{\mathrm{ref}}$")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "alpha_L_vs_sigma_ref.png", dpi=600)
    plt.close()
    print(f"saved: {FIG_DIR}/alpha_L_vs_sigma_ref.png")

    print("\n=== Sensitivity summary ===")
    for sr_key, info in summary.items():
        print(
            f"  σ_ref={sr_key}  α_L overall median={info['alpha_L_median_overall']:.3f}  "
            f"range=[{info['alpha_L_min_overall']:.3f}, {info['alpha_L_max_overall']:.3f}]"
        )


if __name__ == "__main__":
    main()
