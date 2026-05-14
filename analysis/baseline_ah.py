"""Andreasen-Huge style one-step finite-difference local-volatility calibration,
generalised to dividend-aware Dupire equation.

Reference: Andreasen, J. and Huge, B.N. (2011) "Volatility Interpolation",
Risk Magazine March, 76-79. We use the same one-step implicit scheme they
propose, but applied to the *full* Dupire forward equation with drift and
dividend (their original paper assumes r = q = 0):

    ∂C/∂T = (1/2) σ_loc²(K, T) K² ∂²C/∂K² - (r-q) K ∂C/∂K - q C.

Implicit time-step on a uniform K grid is solved via tridiagonal solve.
The local-vol is piecewise-constant in K per maturity slice; the slice's
piecewise levels are calibrated by L-BFGS-B against observed market prices.
Maturity ladder is bootstrapped forward from t = 0.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import solve_banded
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class AHSlice:
    T: float
    boundaries: np.ndarray
    thetas: np.ndarray
    c_at_T: np.ndarray


def _expand_pwc(thetas: np.ndarray, boundaries: np.ndarray, k_grid: np.ndarray) -> np.ndarray:
    """Piecewise-constant theta on the dense k_grid. Segment j covers
    (boundaries[j-1], boundaries[j]] with theta thetas[j]; outside the
    boundary range we extend with the nearest segment's theta."""
    seg = np.searchsorted(boundaries, k_grid, side="left")
    seg = np.clip(seg, 0, len(thetas) - 1)
    return thetas[seg]


def _full_dupire_step(c: np.ndarray, theta_grid: np.ndarray,
                      K_grid: np.ndarray, dt: float, r: float, q: float) -> np.ndarray:
    """One implicit step of (I - dt * L) c_next = c, where
       L = (1/2) σ² K² ∂_KK - (r-q) K ∂_K - q I,
    discretised with central differences on a uniform K grid."""
    n = len(c)
    dK = float(K_grid[1] - K_grid[0])
    K2 = K_grid ** 2
    a = 0.5 * theta_grid ** 2 * K2 / dK ** 2          # diffusion coefficient
    b = (r - q) * K_grid / (2.0 * dK)                  # drift coefficient
    # M = I - dt*L → M c_next = c
    sub = -dt * (a + b)            # M[j, j-1]
    diag = 1.0 + dt * (2.0 * a + q)
    sup = -dt * (a - b)            # M[j, j+1]
    # Boundary: pin c_next[0] = S * exp(-q*T) - K_grid[0] * exp(-r*T) is too involved;
    # instead use C(0, K=K_min) ≈ c[0] (Dirichlet on c) and C(T, K_max) = 0.
    sub[0] = 0.0; diag[0] = 1.0; sup[0] = 0.0
    sub[-1] = 0.0; diag[-1] = 1.0; sup[-1] = 0.0
    rhs = c.copy()
    # banded matrix shape: (3, n) with row 0 = upper, 1 = main, 2 = lower
    ab = np.zeros((3, n))
    ab[0, 1:] = sup[:-1]
    ab[1, :] = diag
    ab[2, :-1] = sub[1:]
    return solve_banded((1, 1), ab, rhs)


def _bs_iv_init_guess(S: float, K: np.ndarray, T: float, r: float, q: float,
                      C_obs: np.ndarray) -> np.ndarray:
    """Return per-strike BS implied vol as initial guess for AH thetas."""
    from scipy.optimize import brentq
    from scipy.stats import norm

    def bs(sig, K_one, T_, S_, r_, q_):
        if sig <= 0 or T_ <= 0:
            return max(S_ - K_one, 0.0)
        d1 = (np.log(S_ / K_one) + (r_ - q_ + 0.5 * sig ** 2) * T_) / (sig * np.sqrt(T_))
        d2 = d1 - sig * np.sqrt(T_)
        return S_ * np.exp(-q_ * T_) * norm.cdf(d1) - K_one * np.exp(-r_ * T_) * norm.cdf(d2)

    out = []
    for k_one, c_one in zip(K, C_obs):
        try:
            iv = brentq(lambda s: bs(s, k_one, T, S, r, q) - c_one, 1e-3, 3.0, xtol=1e-6)
        except Exception:
            iv = 0.3
        out.append(iv)
    return np.clip(np.array(out), 0.1, 1.5)


class AndreasenHugeSurface:
    def __init__(self, df: pd.DataFrame, n_grid: int = 401, K_pad: float = 0.5,
                 theta_bounds: tuple[float, float] = (0.08, 1.5)):
        self.S = float(df["S"].iloc[0])
        self.r = float(df["r"].iloc[0])
        self.q = float(df["q_implied"].iloc[0])
        self.Ts = sorted(df["T"].unique())

        K_min = max(float(df["K"].min()) * (1 - K_pad), 1e-3)
        K_max = float(df["K"].max()) * (1 + K_pad)
        self.K_grid = np.linspace(K_min, K_max, n_grid)
        self.dK = float(self.K_grid[1] - self.K_grid[0])

        # Initial: undiscounted call payoff at t=0
        c = np.maximum(self.S - self.K_grid, 0.0)
        T_prev = 0.0
        self.slices: list[AHSlice] = []

        for T in self.Ts:
            dt = T - T_prev
            sub = df[df["T"] == T].sort_values("K")
            obs_K = sub["K"].to_numpy(dtype=float)
            obs_C = sub["call_equiv"].to_numpy(dtype=float)
            n_seg = len(obs_K)

            theta0 = _bs_iv_init_guess(self.S, obs_K, T, self.r, self.q, obs_C)
            bounds = [theta_bounds] * n_seg

            def loss(theta: np.ndarray, c=c, dt=dt) -> float:
                theta_grid = _expand_pwc(theta, obs_K, self.K_grid)
                c_next = _full_dupire_step(c, theta_grid, self.K_grid, dt, self.r, self.q)
                pred = np.interp(obs_K, self.K_grid, c_next)
                return float(np.mean((pred - obs_C) ** 2))

            res = minimize(loss, theta0, method="L-BFGS-B", bounds=bounds,
                           options={"maxiter": 400, "ftol": 1e-9})
            theta_opt = np.asarray(res.x)
            theta_grid = _expand_pwc(theta_opt, obs_K, self.K_grid)
            c = _full_dupire_step(c, theta_grid, self.K_grid, dt, self.r, self.q)

            self.slices.append(AHSlice(T=float(T), boundaries=obs_K,
                                       thetas=theta_opt, c_at_T=c.copy()))
            T_prev = float(T)

    def C(self, K: float, T: float) -> float:
        if T <= 0:
            return float(np.interp(K, self.K_grid, np.maximum(self.S - self.K_grid, 0.0)))
        idx_below = -1
        for i, s in enumerate(self.slices):
            if s.T <= T:
                idx_below = i
            else:
                break
        if idx_below == -1:
            theta_grid = _expand_pwc(self.slices[0].thetas, self.slices[0].boundaries, self.K_grid)
            c0 = np.maximum(self.S - self.K_grid, 0.0)
            c_t = _full_dupire_step(c0, theta_grid, self.K_grid, T, self.r, self.q)
        elif idx_below == len(self.slices) - 1 and T >= self.slices[-1].T:
            c_t = self.slices[-1].c_at_T
        else:
            s_below = self.slices[idx_below]
            s_above = self.slices[idx_below + 1]
            theta_grid = _expand_pwc(s_above.thetas, s_above.boundaries, self.K_grid)
            dt = T - s_below.T
            c_t = _full_dupire_step(s_below.c_at_T, theta_grid, self.K_grid, dt, self.r, self.q)
        return float(np.interp(K, self.K_grid, c_t))

    def predict(self, K_arr: np.ndarray, T_arr: np.ndarray) -> np.ndarray:
        return np.array([self.C(float(K), float(T)) for K, T in zip(K_arr, T_arr)])


def fit_ah(df: pd.DataFrame) -> AndreasenHugeSurface:
    return AndreasenHugeSurface(df)


def _main() -> None:
    df = pd.read_csv(ROOT / "data" / "clean.csv")
    print(f"loaded clean: N={len(df)}")
    surf = fit_ah(df)
    pred = surf.predict(df["K"].to_numpy(), df["T"].to_numpy())
    err = pred - df["call_equiv"].to_numpy()
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    print(f"\nAH (full Dupire) in-sample  RMSE={rmse:.4f}  MAE={mae:.4f}  max|err|={np.max(np.abs(err)):.4f}")
    for s in surf.slices:
        print(f"  T={s.T:.4f}  n_seg={len(s.thetas)}  theta range=[{s.thetas.min():.3f}, {s.thetas.max():.3f}]")


if __name__ == "__main__":
    _main()
