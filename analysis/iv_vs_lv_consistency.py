"""Cross-comparison of per-option implied volatilities (computed in §2 from
market prices) and the PINN-recovered local-volatility surface evaluated at
the same (K, T) data points.

For each data row, plot iv (per-option) and sigma_loc(K, T) on the same axes
versus K/F. The two surfaces should agree qualitatively (both trace the same
smile shape), but local vol is generally steeper than implied vol on the wings
(Dupire mapping). This figure is the visual consistency check requested in
the reviewer's Minor #7.
"""
from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))

from train_pinn import build_normalizers, make_models, train, sigma_at  # noqa: E402

CLEAN = ROOT / "data" / "clean.csv"
FIG_DIR = ROOT / "figures" / "consistency"
FIG_DIR.mkdir(parents=True, exist_ok=True)
SEED = 42
NUM_ITER = 10_000
N_COLLOC = 256


def main() -> None:
    df = pd.read_csv(CLEAN)
    S = float(df["S"].iloc[0])
    r = float(df["r"].iloc[0])
    q = float(df["q_implied"].iloc[0])

    norm = build_normalizers(df)
    key = jax.random.PRNGKey(SEED)
    ki, kt = jax.random.split(key)
    model = make_models(ki)
    print(f"training PINN ({NUM_ITER} iter) ...")
    model, _ = train(model, df, norm, NUM_ITER, kt, n_colloc=N_COLLOC, verbose=False)
    _, sigma_net = model

    # evaluate sigma_loc at the data points
    Kd = jnp.asarray(df["K"].to_numpy(), dtype=jnp.float32)
    Td = jnp.asarray(df["T"].to_numpy(), dtype=jnp.float32)
    sigma_loc_at_data = np.asarray(jax.vmap(
        lambda k, t: sigma_at(sigma_net, k, t, norm))(Kd, Td))

    df = df.assign(sigma_loc=sigma_loc_at_data)

    # plot per maturity
    fig, ax = plt.subplots(figsize=(11, 6))
    cmap = plt.get_cmap("viridis")
    Ts = sorted(df["T"].unique())
    for i, T in enumerate(Ts):
        sub = df[df["T"] == T]
        F = S * np.exp((r - q) * T)
        kf = sub["K"].to_numpy() / F
        ax.scatter(kf, sub["iv"], color=cmap(i / max(1, len(Ts) - 1)),
                   marker="o", s=24, alpha=0.7, label=f"σ_iv  T={T:.3f}" if i == 0 else None)
        ax.scatter(kf, sub["sigma_loc"], color=cmap(i / max(1, len(Ts) - 1)),
                   marker="x", s=30, alpha=0.85, label=f"σ_loc T={T:.3f}" if i == 0 else None)
    ax.set_xlabel("K/F")
    ax.set_ylabel("vol")
    ax.set_title(r"Per-option implied vol $\sigma_{iv}$ (○) vs PINN-recovered $\sigma_{loc}$ (×) at data points")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "iv_vs_loc_consistency.png", dpi=140)
    plt.close()
    print(f"saved: {FIG_DIR / 'iv_vs_loc_consistency.png'}")

    # Summary statistic
    diff = df["sigma_loc"] - df["iv"]
    print(f"\nsigma_loc - sigma_iv statistics (at data points, N={len(df)}):")
    print(f"  mean={diff.mean():+.4f}  std={diff.std():.4f}  "
          f"corr={df[['iv', 'sigma_loc']].corr().iloc[0, 1]:.3f}")


if __name__ == "__main__":
    main()
