"""Baselines for comparison with the Dupire PINN.

Implements:
  1. Raw SVI (Gatheral) per-maturity slice fit on total implied variance
  2. Cubic spline smoothing per-maturity slice on call_equiv prices

Both produce a callable surface  predict(K, T) -> C  for evaluation.
For T not at a fitted maturity, SVI linearly interpolates parameters in T;
the cubic spline baseline linearly interpolates prices in T.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize
from scipy.stats import norm

ROOT = Path(__file__).resolve().parent.parent
CLEAN = ROOT / "data" / "clean.csv"
OUT_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures" / "baselines"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# Raw SVI (Gatheral)
# --------------------------------------------------------------------------
@dataclass
class SVIParams:
    a: float
    b: float
    rho: float
    m: float
    s: float

    def w(self, k: np.ndarray) -> np.ndarray:
        """Total implied variance at log-moneyness k."""
        return self.a + self.b * (self.rho * (k - self.m) + np.sqrt((k - self.m) ** 2 + self.s ** 2))


def fit_svi_slice(k: np.ndarray, w_obs: np.ndarray) -> SVIParams:
    """Fit raw SVI to one maturity slice (k, w_obs)."""
    # Initial guess: a near min(w), b small, rho=-0.3, m=mean(k), s small
    x0 = [float(np.min(w_obs) * 0.5), 0.1, -0.3, float(np.mean(k)), 0.1]
    bounds = [(-1.0, 1.0), (0.0, 5.0), (-0.999, 0.999), (-3.0, 3.0), (1e-4, 5.0)]

    def loss(theta):
        a, b, rho, m, s = theta
        w = a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + s ** 2))
        return float(np.mean((w - w_obs) ** 2))

    # No-negative-vol constraint: a + b*s*sqrt(1-rho^2) >= 0
    cons = ({"type": "ineq", "fun": lambda th: th[0] + th[1] * th[4] * np.sqrt(1 - th[2] ** 2)},)

    res = minimize(
        loss, x0, method="SLSQP", bounds=bounds, constraints=cons,
        options={"maxiter": 500, "ftol": 1e-10},
    )
    a, b, rho, m, s = res.x
    return SVIParams(a=float(a), b=float(b), rho=float(rho), m=float(m), s=float(s))


def black_call(F: float, K: float, w: float, T: float, r: float) -> float:
    """Black-76 call (forward formula). w = total implied variance σ²T."""
    if w <= 0 or T <= 0:
        return float(np.exp(-r * T) * max(F - K, 0.0))
    sqrt_w = np.sqrt(w)
    d1 = (np.log(F / K) + 0.5 * w) / sqrt_w
    d2 = d1 - sqrt_w
    return float(np.exp(-r * T) * (F * norm.cdf(d1) - K * norm.cdf(d2)))


# --------------------------------------------------------------------------
# Surface objects
# --------------------------------------------------------------------------
class SVISurface:
    """SVI per-slice fit, with linear-in-T interpolation of params for queries
    at intermediate maturities."""

    def __init__(self, params_by_T: dict[float, SVIParams], S: float, r: float, q: float):
        self.Ts = np.array(sorted(params_by_T.keys()))
        self.params = [params_by_T[T] for T in self.Ts]
        self.S = S
        self.r = r
        self.q = q

    def _interp_params(self, T: float) -> SVIParams:
        if T <= self.Ts[0]:
            return self.params[0]
        if T >= self.Ts[-1]:
            return self.params[-1]
        i = int(np.searchsorted(self.Ts, T))
        T0, T1 = self.Ts[i - 1], self.Ts[i]
        w_ = (T - T0) / (T1 - T0)
        p0, p1 = self.params[i - 1], self.params[i]
        return SVIParams(
            a=(1 - w_) * p0.a + w_ * p1.a,
            b=(1 - w_) * p0.b + w_ * p1.b,
            rho=(1 - w_) * p0.rho + w_ * p1.rho,
            m=(1 - w_) * p0.m + w_ * p1.m,
            s=(1 - w_) * p0.s + w_ * p1.s,
        )

    def C(self, K: float, T: float) -> float:
        F = self.S * np.exp((self.r - self.q) * T)
        k = np.log(K / F)
        p = self._interp_params(T)
        w = float(p.w(np.array([k]))[0])
        w = max(w, 1e-8)
        return black_call(F, K, w, T, self.r)

    def predict(self, K_arr: np.ndarray, T_arr: np.ndarray) -> np.ndarray:
        return np.array([self.C(float(K), float(T)) for K, T in zip(K_arr, T_arr)])


class SplineSurface:
    """Cubic spline per maturity slice on (K, call_equiv).
    Linear-in-T interpolation between slices for queries at intermediate T."""

    def __init__(self, splines_by_T: dict[float, CubicSpline]):
        self.Ts = np.array(sorted(splines_by_T.keys()))
        self.splines = [splines_by_T[T] for T in self.Ts]

    def C(self, K: float, T: float) -> float:
        if T <= self.Ts[0]:
            return float(self.splines[0](K))
        if T >= self.Ts[-1]:
            return float(self.splines[-1](K))
        i = int(np.searchsorted(self.Ts, T))
        T0, T1 = self.Ts[i - 1], self.Ts[i]
        w = (T - T0) / (T1 - T0)
        return float((1 - w) * self.splines[i - 1](K) + w * self.splines[i](K))

    def predict(self, K_arr: np.ndarray, T_arr: np.ndarray) -> np.ndarray:
        return np.array([self.C(float(K), float(T)) for K, T in zip(K_arr, T_arr)])


# --------------------------------------------------------------------------
# Fitting drivers
# --------------------------------------------------------------------------
def fit_svi(df: pd.DataFrame) -> SVISurface:
    S = float(df["S"].iloc[0])
    r = float(df["r"].iloc[0])
    q = float(df["q_implied"].iloc[0])
    params_by_T = {}
    for T, g in df.groupby("T"):
        F = S * np.exp((r - q) * T)
        k = np.log(g["K"].to_numpy() / F)
        w_obs = (g["iv"].to_numpy() ** 2) * T
        params_by_T[float(T)] = fit_svi_slice(k, w_obs)
    return SVISurface(params_by_T, S=S, r=r, q=q)


def fit_spline(df: pd.DataFrame) -> SplineSurface:
    splines_by_T = {}
    for T, g in df.groupby("T"):
        gs = g.sort_values("K")
        K_arr = gs["K"].to_numpy(dtype=float)
        C_arr = gs["call_equiv"].to_numpy(dtype=float)
        # de-duplicate any tied K (shouldn't happen but defensive)
        _, idx = np.unique(K_arr, return_index=True)
        K_arr, C_arr = K_arr[idx], C_arr[idx]
        if len(K_arr) >= 4:
            splines_by_T[float(T)] = CubicSpline(K_arr, C_arr, bc_type="natural", extrapolate=True)
    return SplineSurface(splines_by_T)


# --------------------------------------------------------------------------
# Eval & plot
# --------------------------------------------------------------------------
def in_sample_metrics(surface, df: pd.DataFrame) -> dict:
    pred = surface.predict(df["K"].to_numpy(), df["T"].to_numpy())
    err = pred - df["call_equiv"].to_numpy()
    return {
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "max_abs_err": float(np.max(np.abs(err))),
    }


def main() -> None:
    df = pd.read_csv(CLEAN)
    print(f"loaded clean data: {len(df)} rows")

    print("\n[SVI] fitting per-maturity slices ...")
    svi = fit_svi(df)
    svi_metrics = in_sample_metrics(svi, df)
    print(f"  in-sample RMSE: ${svi_metrics['rmse']:.4f}  MAE: ${svi_metrics['mae']:.4f}  max|err|: ${svi_metrics['max_abs_err']:.4f}")

    print("\n[Spline] fitting per-maturity natural cubic splines ...")
    spline = fit_spline(df)
    spline_metrics = in_sample_metrics(spline, df)
    print(f"  in-sample RMSE: ${spline_metrics['rmse']:.4f}  MAE: ${spline_metrics['mae']:.4f}  max|err|: ${spline_metrics['max_abs_err']:.4f}")

    # save SVI params
    svi_params_dump = {
        f"{T:.6f}": asdict(svi.params[i]) for i, T in enumerate(svi.Ts)
    }
    with open(OUT_DIR / "svi_params.json", "w") as f:
        json.dump(svi_params_dump, f, indent=2)
    print(f"\nsaved: {OUT_DIR / 'svi_params.json'}")

    # plot smile fits per maturity
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.get_cmap("viridis")
    Ts = sorted(df["T"].unique())
    S = float(df["S"].iloc[0])
    r = float(df["r"].iloc[0])
    q = float(df["q_implied"].iloc[0])
    for i, T in enumerate(Ts):
        g = df[df["T"] == T]
        F = S * np.exp((r - q) * T)
        k = np.log(g["K"].to_numpy() / F)
        ax.scatter(k, g["iv"], color=cmap(i / max(1, len(Ts) - 1)), s=24, alpha=0.7)
        # SVI fit curve
        kk = np.linspace(k.min() - 0.05, k.max() + 0.05, 100)
        p = svi.params[i]
        w = p.w(kk)
        iv_fit = np.sqrt(np.maximum(w / T, 1e-8))
        ax.plot(kk, iv_fit, color=cmap(i / max(1, len(Ts) - 1)), lw=1.5, label=f"T={T:.3f}")
    ax.set_xlabel("log-moneyness  k = log(K/F)")
    ax.set_ylabel("implied vol")
    ax.set_title("SVI per-maturity slice fits (lines) vs market IV (dots)")
    ax.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "svi_slices.png", dpi=600)
    plt.close()
    print(f"saved: {FIG_DIR / 'svi_slices.png'}")

    # combined metrics summary
    summary = {
        "svi": svi_metrics,
        "spline": spline_metrics,
        "n_data": len(df),
        "n_maturities": len(Ts),
    }
    with open(OUT_DIR / "baseline_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"saved: {OUT_DIR / 'baseline_metrics.json'}")


if __name__ == "__main__":
    main()
