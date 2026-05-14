"""Dupire PINN for local-volatility surface calibration on TSM options.

Two equinox MLPs:
  C(K, T)         -- call price surface
  sigma(K, T)     -- local volatility surface

Coupled via the dividend-aware Dupire equation:
  ∂_T C = (1/2) σ² K² ∂_KK C - (r-q) K ∂_K C - q C

Arbitrage soft constraints (penalties):
  ∂_T C  ≥ 0      calendar-spread monotonicity
  ∂_KK C ≥ 0      butterfly convexity
  ∂_K C  ≤ 0      call decreasing in strike

Both nets use tanh activation (smooth, required for second derivative).
Inputs (K, T) are standardized; derivatives are taken in *physical* units
(JAX chain rule handles the linear normalization automatically).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optax
import pandas as pd

# --------------------------------------------------------------------------
# Paths and hyperparameters
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
CLEAN_CSV = ROOT / "data" / "clean.csv"
OUT_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures" / "pinn"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
NUM_ITER = 10_000
LR_INIT = 1e-3
N_COLLOC = 256
PRINT_EVERY = 500
SIGMA_REF = 0.35  # BS baseline volatility — chosen near observed ATM IV
C_DEPTH, C_WIDTH = 4, 64
SIGMA_DEPTH, SIGMA_WIDTH = 4, 32

# Loss weights (tuned to roughly balance scales after first pass)
W_DATA = 1.0
W_PDE = 0.01
W_ARB = 0.05
PDE_WARMUP_FRAC = 0.20  # ramp PDE/arbitrage weight from 0 to full over first 20% of iters

# Collocation sampling box (extends beyond data for boundary behavior)
K_PAD = 0.10  # 10% padding outside data range
T_PAD = 0.0   # do not extrapolate in time


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def load_data() -> pd.DataFrame:
    df = pd.read_csv(CLEAN_CSV)
    return df


def build_normalizers(df: pd.DataFrame) -> dict:
    return dict(
        K_mean=float(df["K"].mean()),
        K_std=float(df["K"].std()),
        T_mean=float(df["T"].mean()),
        T_std=float(df["T"].std()),
        S=float(df["S"].iloc[0]),
        r=float(df["r"].iloc[0]),
        q=float(df["q_implied"].iloc[0]),
    )


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
def make_models(key):
    k1, k2 = jax.random.split(key, 2)
    c_net = eqx.nn.MLP(
        in_size=2, out_size="scalar",
        width_size=C_WIDTH, depth=C_DEPTH,
        activation=jnp.tanh, key=k1,
    )
    sigma_net = eqx.nn.MLP(
        in_size=2, out_size="scalar",
        width_size=SIGMA_WIDTH, depth=SIGMA_DEPTH,
        activation=jnp.tanh, key=k2,
    )
    return c_net, sigma_net


def C_at(c_net, K, T, norm):
    """Scalar C(K, T). K, T in physical units.

    Parameterization:  C = BS(K, T; σ_ref) + softplus(NN(K, T))
    The Black-Scholes baseline at a fixed reference vol σ_ref is C^∞ smooth
    everywhere (no kink at K=F), giving the σ network a well-conditioned
    second-derivative landscape. The NN learns the smile residual on top.
    """
    S = norm["S"]; r = norm["r"]; q = norm["q"]
    F = S * jnp.exp((r - q) * T)
    sqrt_T = jnp.sqrt(T + 1e-8)
    d1 = (jnp.log(F / K) + 0.5 * SIGMA_REF**2 * T) / (SIGMA_REF * sqrt_T)
    d2 = d1 - SIGMA_REF * sqrt_T
    bs = jnp.exp(-r * T) * (F * jax.scipy.stats.norm.cdf(d1) - K * jax.scipy.stats.norm.cdf(d2))
    Kn = (K - norm["K_mean"]) / norm["K_std"]
    Tn = (T - norm["T_mean"]) / norm["T_std"]
    x = jnp.stack([Kn, Tn])
    time_value = jax.nn.softplus(c_net(x))
    return bs + time_value


def sigma_at(sigma_net, K, T, norm):
    Kn = (K - norm["K_mean"]) / norm["K_std"]
    Tn = (T - norm["T_mean"]) / norm["T_std"]
    x = jnp.stack([Kn, Tn])
    return jax.nn.softplus(sigma_net(x))


# --------------------------------------------------------------------------
# Loss
# --------------------------------------------------------------------------
def make_loss_fn(norm):
    """Closure-bound loss; norm is captured (static).

    Computes C and its (∂_K, ∂_T, ∂²_KK) in a single AD pass per collocation
    point via grad + hessian on a 2-vector input. Much cheaper than nested
    jax.grad through closures.
    """
    r = norm["r"]
    q = norm["q"]
    K_std = norm["K_std"]
    T_std = norm["T_std"]
    K_mean = norm["K_mean"]
    T_mean = norm["T_mean"]

    S_spot = norm["S"]

    def c_eval(c_net, kt_phys):
        K = kt_phys[0]
        T = kt_phys[1]
        # BS baseline at σ_ref — smooth in (K, T), no kinks
        F = S_spot * jnp.exp((r - q) * T)
        sqrt_T = jnp.sqrt(T + 1e-8)
        d1 = (jnp.log(F / K) + 0.5 * SIGMA_REF**2 * T) / (SIGMA_REF * sqrt_T)
        d2 = d1 - SIGMA_REF * sqrt_T
        bs = jnp.exp(-r * T) * (F * jax.scipy.stats.norm.cdf(d1) - K * jax.scipy.stats.norm.cdf(d2))
        kt_norm = jnp.stack([
            (K - K_mean) / K_std,
            (T - T_mean) / T_std,
        ])
        return bs + jax.nn.softplus(c_net(kt_norm))

    def sigma_eval(sigma_net, kt_phys):
        kt_norm = jnp.stack([
            (kt_phys[0] - K_mean) / K_std,
            (kt_phys[1] - T_mean) / T_std,
        ])
        return jax.nn.softplus(sigma_net(kt_norm))

    def C_and_derivs(c_net, K, T):
        kt = jnp.stack([K, T])
        f = lambda x: c_eval(c_net, x)
        C = f(kt)
        g = jax.grad(f)(kt)              # ∂/∂K, ∂/∂T (physical units)
        h = jax.hessian(f)(kt)           # 2x2 (physical units)
        return C, g[0], g[1], h[0, 0]

    def per_point(c_net, sigma_net, K, T):
        C, dCdK, dCdT, d2CdK2 = C_and_derivs(c_net, K, T)
        sigma = sigma_eval(sigma_net, jnp.stack([K, T]))
        pde_res = dCdT - 0.5 * sigma**2 * K**2 * d2CdK2 + (r - q) * K * dCdK + q * C
        arb_T = jax.nn.relu(-dCdT) ** 2     # calendar: ∂T C ≥ 0
        arb_KK = jax.nn.relu(-d2CdK2) ** 2  # butterfly: ∂KK C ≥ 0
        arb_K = jax.nn.relu(dCdK) ** 2      # K-monotone: ∂K C ≤ 0
        return pde_res ** 2, arb_T, arb_KK, arb_K

    def C_only(c_net, K, T):
        return c_eval(c_net, jnp.stack([K, T]))

    def loss_fn(model, data_K, data_T, data_C, colloc_K, colloc_T, w_pde, w_arb):
        c_net, sigma_net = model

        c_pred = jax.vmap(C_only, in_axes=(None, 0, 0))(c_net, data_K, data_T)
        L_data = jnp.mean((c_pred - data_C) ** 2)

        pde2, aT, aKK, aK = jax.vmap(
            per_point, in_axes=(None, None, 0, 0)
        )(c_net, sigma_net, colloc_K, colloc_T)
        L_pde = jnp.mean(pde2)
        L_arb = jnp.mean(aT) + jnp.mean(aKK) + jnp.mean(aK)

        total = W_DATA * L_data + w_pde * L_pde + w_arb * L_arb
        return total, (L_data, L_pde, L_arb)

    return loss_fn


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
def train(
    model, data, norm, num_iter, key,
    *,
    n_colloc: int = N_COLLOC,
    lr_init: float = LR_INIT,
    w_pde_max: float = W_PDE,
    w_arb_max: float = W_ARB,
    pde_warmup_frac: float = PDE_WARMUP_FRAC,
    k_pad: float = K_PAD,
    t_pad: float = T_PAD,
    print_every: int = PRINT_EVERY,
    verbose: bool = True,
):
    data_K = jnp.asarray(data["K"].to_numpy(), dtype=jnp.float32)
    data_T = jnp.asarray(data["T"].to_numpy(), dtype=jnp.float32)
    data_C = jnp.asarray(data["call_equiv"].to_numpy(), dtype=jnp.float32)

    K_lo = float(data["K"].min()) * (1 - k_pad)
    K_hi = float(data["K"].max()) * (1 + k_pad)
    T_lo = float(data["T"].min()) * (1 - t_pad) if t_pad > 0 else float(data["T"].min())
    T_hi = float(data["T"].max())

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=lr_init,
        warmup_steps=int(0.05 * num_iter),
        decay_steps=num_iter,
        end_value=lr_init * 0.01,
    )
    optimizer = optax.adam(schedule)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    loss_fn = make_loss_fn(norm)

    @eqx.filter_jit
    def step_fn(model, opt_state, dK, dT, dC, cK, cT, w_pde, w_arb):
        (loss, aux), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
            model, dK, dT, dC, cK, cT, w_pde, w_arb
        )
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss, aux

    history = []
    t0 = time.time()
    warmup = max(1, int(num_iter * pde_warmup_frac))
    if verbose:
        print(f"[t=0.0s] entering training loop, JIT compile begins on first step ...", flush=True)
    for it in range(num_iter):
        key, sub = jax.random.split(key)
        u = jax.random.uniform(sub, (n_colloc, 2))
        colloc_K = u[:, 0] * (K_hi - K_lo) + K_lo
        colloc_T = u[:, 1] * (T_hi - T_lo) + T_lo

        ramp = min(1.0, it / warmup)
        w_pde = jnp.asarray(w_pde_max * ramp, dtype=jnp.float32)
        w_arb = jnp.asarray(w_arb_max * ramp, dtype=jnp.float32)

        model, opt_state, loss, aux = step_fn(
            model, opt_state, data_K, data_T, data_C, colloc_K, colloc_T, w_pde, w_arb
        )
        l_data, l_pde, l_arb = aux
        if it == 0 and verbose:
            elapsed = time.time() - t0
            print(f"[t={elapsed:.1f}s] first iter complete (JIT done)", flush=True)

        if it % print_every == 0 or it == num_iter - 1:
            history.append({
                "iter": it,
                "loss": float(loss),
                "L_data": float(l_data),
                "L_pde": float(l_pde),
                "L_arb": float(l_arb),
                "w_pde": float(w_pde),
                "w_arb": float(w_arb),
            })
            if verbose:
                elapsed = time.time() - t0
                print(
                    f"iter {it:>6d}  total={float(loss):.4e}  "
                    f"data={float(l_data):.4e}  "
                    f"pde={float(l_pde):.4e}  "
                    f"arb={float(l_arb):.4e}  "
                    f"({elapsed:.1f}s)"
                )

    return model, pd.DataFrame(history)


# --------------------------------------------------------------------------
# Diagnostics & plotting
# --------------------------------------------------------------------------
def evaluate(model, data, norm):
    c_net, sigma_net = model
    K = jnp.asarray(data["K"].to_numpy(), dtype=jnp.float32)
    T = jnp.asarray(data["T"].to_numpy(), dtype=jnp.float32)
    Cobs = data["call_equiv"].to_numpy()
    C_pred = np.asarray(jax.vmap(lambda k, t: C_at(c_net, k, t, norm))(K, T))
    err = C_pred - Cobs
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    return dict(rmse_in_sample=rmse, mae_in_sample=mae)


def plot_loss(history, out_path):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.semilogy(history["iter"], history["L_data"], label="data")
    ax.semilogy(history["iter"], history["L_pde"], label="pde")
    ax.semilogy(history["iter"], history["L_arb"], label="arbitrage")
    ax.semilogy(history["iter"], history["loss"], label="total", lw=2, color="k")
    ax.set_xlabel("iteration")
    ax.set_ylabel("loss (log)")
    ax.set_title("PINN training history")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=600)
    plt.close()


def plot_surfaces(model, data, norm, out_dir):
    c_net, sigma_net = model
    Kg = np.linspace(data["K"].min() * 0.95, data["K"].max() * 1.05, 80)
    Tg = np.linspace(data["T"].min(), data["T"].max(), 50)
    KK, TT = np.meshgrid(Kg, Tg)
    pts_K = jnp.asarray(KK.ravel(), dtype=jnp.float32)
    pts_T = jnp.asarray(TT.ravel(), dtype=jnp.float32)
    C_grid = np.asarray(jax.vmap(lambda k, t: C_at(c_net, k, t, norm))(pts_K, pts_T)).reshape(KK.shape)
    S_grid = np.asarray(jax.vmap(lambda k, t: sigma_at(sigma_net, k, t, norm))(pts_K, pts_T)).reshape(KK.shape)

    # C surface vs data
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    cs = ax[0].contourf(KK, TT, C_grid, levels=20, cmap="viridis")
    ax[0].scatter(data["K"], data["T"], c=data["call_equiv"],
                  edgecolor="k", s=20, cmap="viridis")
    plt.colorbar(cs, ax=ax[0], label="C")
    ax[0].set_title("C(K, T) surface")
    ax[0].set_xlabel("K")
    ax[0].set_ylabel("T")

    cs2 = ax[1].contourf(KK, TT, S_grid, levels=20, cmap="plasma")
    plt.colorbar(cs2, ax=ax[1], label="σ_loc")
    ax[1].set_title("σ_loc(K, T) — local volatility surface")
    ax[1].set_xlabel("K")
    ax[1].set_ylabel("T")
    plt.tight_layout()
    plt.savefig(out_dir / "surfaces.png", dpi=600)
    plt.close()

    # residuals at data points
    Kd = jnp.asarray(data["K"].to_numpy(), dtype=jnp.float32)
    Td = jnp.asarray(data["T"].to_numpy(), dtype=jnp.float32)
    C_pred = np.asarray(jax.vmap(lambda k, t: C_at(c_net, k, t, norm))(Kd, Td))
    resid = C_pred - data["call_equiv"].to_numpy()

    fig, ax = plt.subplots(figsize=(8, 5))
    sc = ax.scatter(data["K"] / norm["S"], data["T"], c=resid,
                    cmap="coolwarm", vmin=-np.abs(resid).max(), vmax=np.abs(resid).max(), s=30)
    plt.colorbar(sc, ax=ax, label="residual ($)")
    ax.set_xlabel("K/S")
    ax.set_ylabel("T")
    ax.set_title("In-sample residuals C_pred - C_obs")
    plt.tight_layout()
    plt.savefig(out_dir / "residuals.png", dpi=600)
    plt.close()


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    df = load_data()
    norm = build_normalizers(df)
    print("normalization:", {k: round(v, 4) for k, v in norm.items()})

    key = jax.random.PRNGKey(SEED)
    key_init, key_train = jax.random.split(key)
    model = make_models(key_init)

    print(f"\ntraining: {NUM_ITER} iterations, N_data={len(df)}, N_colloc={N_COLLOC}/iter")
    model, history = train(model, df, norm, NUM_ITER, key_train)

    metrics = evaluate(model, df, norm)
    print(f"\nin-sample RMSE: ${metrics['rmse_in_sample']:.4f}")
    print(f"in-sample MAE:  ${metrics['mae_in_sample']:.4f}")

    # save
    eqx.tree_serialise_leaves(OUT_DIR / "pinn_model.eqx", model)
    history.to_csv(OUT_DIR / "pinn_history.csv", index=False)
    with open(OUT_DIR / "pinn_metrics.json", "w") as f:
        json.dump(metrics | {"seed": SEED, "num_iter": NUM_ITER}, f, indent=2)

    plot_loss(history, FIG_DIR / "loss_curves.png")
    plot_surfaces(model, df, norm, FIG_DIR)
    print(f"\nsaved: {OUT_DIR / 'pinn_model.eqx'}")
    print(f"saved: {OUT_DIR / 'pinn_history.csv'}")
    print(f"saved: {FIG_DIR}/*.png")


if __name__ == "__main__":
    main()
