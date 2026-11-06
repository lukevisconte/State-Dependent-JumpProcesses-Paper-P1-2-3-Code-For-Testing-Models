from __future__ import annotations

import itertools
import json
import sys
from collections import deque

import numpy as np
import pandas as pd
from scipy import optimize, special, stats

from common import (DT, RESULTS, diagnostics, info_criteria, jump_nmax, load_fit, load_returns,
                    load_warmup_prices, norm_logpdf, save_fit, simulate_level_stats, z_from_tails)
from extensions.heston import V_FLOOR, VOL_HI, VOL_LO

# Supported extension: Bates with a drawdown-barrier jump intensity and jump size.
#   dS_t / S_t- = (mu - lambda_t kbar_t) dt + sqrt(v_t) dW^S_t + (J_t - 1) dN_t
#   dv_t        = kappa (theta - v_t) dt + xi sqrt(v_t) dW^v_t,   d<W^S, W^v> = rho dt
#   B_{t-1}     = c x max(S over the 252 trading days ending at t-1),  c = 0.8 (fixed)
#   d_t         = max(0, ln S_{t-1} - ln B_{t-1})        log distance above the barrier
#   g_t         = exp(-gamma d_t^2)                      one-sided kernel, = 1 at or below B
#   lambda_t    = lambda0 + a g_t                        jump frequency
#   mu_J,t      = muJ0 + b g_t,  ln J_t ~ N(mu_J,t, sJ^2) mean log jump size
#   kbar_t      = exp(mu_J,t + sJ^2/2) - 1               compensator
# Constraints a >= 0, b <= 0, gamma > 0. Everything uses yesterday's close, so lambda_t and mu_J,t
# are known before day t's return. With a = b = 0 the model is exactly Bates (the nested test).
BARRIER_C = 0.8
BARRIER_WINDOW = 252


def barrier_frame(sample_index: pd.DatetimeIndex, c: float = BARRIER_C) -> pd.DataFrame:
    # For each return date t: S_{t-1}, B_{t-1}, the trailing high, the drawdown and d_t. The
    # warm-up prices from 2015-08-01 make the 252-day high defined from the first sample day.
    w = load_warmup_prices()
    high = w.rolling(BARRIER_WINDOW, min_periods=BARRIER_WINDOW).max()
    out = pd.DataFrame({"S_prev": w.shift(1), "B_prev": (c * high).shift(1),
                        "high_prev": high.shift(1)}).loc[sample_index]
    if out.isna().any().any():
        raise ValueError("barrier warm-up too short for the first sample date")
    out["drawdown_prev"] = out["S_prev"] / out["high_prev"] - 1
    out["d"] = np.maximum(0.0, np.log(out["S_prev"]) - np.log(out["B_prev"]))
    return out


def jump_paths(p, d):
    g = np.exp(-p["gamma"] * d ** 2)
    return g, p["lam0"] + p["a"] * g, p["muJ0"] + p["b"] * g


# Grid filter from heston.py with a day-specific jump intensity lambda_t and mean jump size mu_J,t.
def grid_filter_tv(p, r, d, M, want_pit):
    mu, kappa, theta, xi, rho, sJ = p["mu"], p["kappa"], p["theta"], p["xi"], p["rho"], p["sJ"]
    _, lam_t, muJ_t = jump_paths(p, d)
    kbar_t = np.exp(muJ_t + 0.5 * sJ ** 2) - 1

    edges = np.linspace(VOL_LO, VOL_HI, M + 1)
    vc = (0.5 * (edges[1:] + edges[:-1])) ** 2
    Bd = edges[1:-1] ** 2
    vd = vc * DT
    sdd = np.sqrt(vd)
    drift_v = vc + kappa * (theta - vc) * DT

    cdf = stats.gamma.cdf(edges ** 2, 2 * kappa * theta / xi ** 2, scale=xi ** 2 / (2 * kappa))
    cdf[0], cdf[-1] = 0.0, 1.0
    pi = np.maximum(np.diff(cdf), 0.0)
    pi /= pi.sum()

    n_grid = np.arange(jump_nmax(lam_t.max() * DT) + 1)
    jv = n_grid[None, :] * sJ ** 2
    gain = vd[:, None] / (vd[:, None] + jv)
    ratio_var = jv / (vd[:, None] + jv)
    sq2 = 1 - rho ** 2

    T = len(r)
    loglik = 0.0
    u_out = np.empty(T) if want_pit else None
    s_out = np.empty(T) if want_pit else None
    for t in range(T):
        rt = r[t]
        logpn = stats.poisson.logpmf(n_grid, lam_t[t] * DT)
        m = (mu - 0.5 * vc - lam_t[t] * kbar_t[t]) * DT
        mn = m[:, None] + n_grid[None, :] * muJ_t[t]
        vn = vd[:, None] + jv
        comp = logpn[None, :] + norm_logpdf(rt, mn, vn)
        cm = comp.max(axis=1, keepdims=True)
        ec = np.exp(comp - cm)
        logf = cm[:, 0] + np.log(ec.sum(axis=1))
        wn = ec / ec.sum(axis=1, keepdims=True)
        if want_pit:
            pn = np.exp(logpn)
            xx = (rt - mn) / np.sqrt(vn)
            u_out[t], s_out[t] = pi @ (special.ndtr(xx) @ pn), pi @ (special.ndtr(-xx) @ pn)
        cmax = logf.max()
        f = np.exp(logf - cmax)
        lt = pi @ f
        loglik += cmax + np.log(lt)
        post = pi * f / lt
        new = np.zeros(M)
        for k in range(len(n_grid)):
            q = post * wn[:, k]
            keep = q > 1e-14
            if not keep.any():
                continue
            e_mean = gain[keep, k] * (rt - mn[keep, k]) / sdd[keep]
            mean_v = drift_v[keep] + xi * sdd[keep] * rho * e_mean
            sd_v = xi * sdd[keep] * np.sqrt(sq2 + rho ** 2 * ratio_var[keep, k])
            C = special.ndtr((Bd[None, :] - mean_v[:, None]) / sd_v[:, None])
            K = np.diff(np.concatenate([np.zeros((C.shape[0], 1)), C, np.ones((C.shape[0], 1))], axis=1), axis=1)
            new += q[keep] @ K
        pi = new / new.sum()
    return loglik, (u_out, s_out)


# Particle filter with day-specific jumps: the reported log-likelihood, comparable with Bates.
def particle_filter_tv(p, r, d, n_particles, seed):
    mu, kappa, theta, xi, rho, sJ = p["mu"], p["kappa"], p["theta"], p["xi"], p["rho"], p["sJ"]
    _, lam_t, muJ_t = jump_paths(p, d)
    kbar_t = np.exp(muJ_t + 0.5 * sJ ** 2) - 1
    T = len(r)
    rng = np.random.default_rng(seed)
    v = np.maximum(rng.gamma(2 * kappa * theta / xi ** 2, xi ** 2 / (2 * kappa), n_particles), V_FLOOR)
    U_res = rng.random(T)
    nmax = jump_nmax(lam_t.max() * DT)
    n_grid = np.arange(nmax + 1)
    sq = np.sqrt(1 - rho ** 2)
    idx_base = np.arange(n_particles)
    loglik = 0.0
    for t in range(T):
        rt = r[t]
        logpn = stats.poisson.logpmf(n_grid, lam_t[t] * DT)
        m = (mu - 0.5 * v - lam_t[t] * kbar_t[t]) * DT
        vd = v * DT
        mn = m[:, None] + n_grid[None, :] * muJ_t[t]
        vn = vd[:, None] + n_grid[None, :] * sJ ** 2
        comp = logpn[None, :] + norm_logpdf(rt, mn, vn)
        cm = comp.max(axis=1, keepdims=True)
        logf = cm[:, 0] + np.log(np.exp(comp - cm).sum(axis=1))
        cmax = logf.max()
        w = np.exp(logf - cmax)
        wsum = w.sum()
        loglik += cmax + np.log(wsum / n_particles)
        order = np.argsort(v)
        cw = np.cumsum(w[order] / wsum)
        cw[-1] = 1.0
        sel = order[np.searchsorted(cw, (U_res[t] + idx_base) / n_particles)]
        v_r, m_r, vd_r = v[sel], m[sel], vd[sel]
        comp_r = comp[sel]
        post = np.exp(comp_r - comp_r.max(axis=1, keepdims=True))
        post /= post.sum(axis=1, keepdims=True)
        ncount = np.minimum((np.cumsum(post, axis=1) < rng.random(n_particles)[:, None]).sum(axis=1), nmax)
        jvar = ncount * sJ ** 2
        gain = vd_r / (vd_r + jvar)
        D = m_r + gain * (rt - m_r - ncount * muJ_t[t]) + np.sqrt(vd_r * jvar / (vd_r + jvar)) * rng.standard_normal(n_particles)
        eps1 = (D - m_r) / np.sqrt(vd_r)
        v = np.maximum(v_r + kappa * (theta - v_r) * DT + xi * np.sqrt(vd_r) * (rho * eps1 + sq * rng.standard_normal(n_particles)), V_FLOOR)
    return loglik


class BarrierBates:
    name = "BarrierBates"
    param_names = ["mu", "kappa", "theta", "xi", "rho", "lam0", "muJ0", "sJ", "a", "b", "gamma"]
    GRID_M = 150

    # optimiser coordinates: [mu, ln kappa, ln theta, ln xi, atanh rho, ln lam0, muJ0, ln sJ, a, b, ln gamma]
    @staticmethod
    def _unpack(x):
        return {"mu": x[0], "kappa": np.exp(x[1]), "theta": np.exp(x[2]), "xi": np.exp(x[3]),
                "rho": np.tanh(x[4]), "lam0": np.exp(x[5]), "muJ0": x[6], "sJ": np.exp(x[7]),
                "a": x[8], "b": x[9], "gamma": np.exp(x[10])}

    @staticmethod
    def _pack(p):
        return [p["mu"], np.log(p["kappa"]), np.log(p["theta"]), np.log(p["xi"]), np.arctanh(p["rho"]),
                np.log(p["lam0"]), p["muJ0"], np.log(p["sJ"]), p["a"], p["b"], np.log(p["gamma"])]

    # box bounds for L-BFGS-B; a >= 0, b <= 0 and gamma > 0 are enforced here
    BOUNDS = [(-1, 1), (-3, 4.5), (-7, 0), (-5, 2), (-3, 3), (-4, 6), (-0.3, 0.3), (-7, -0.5),
              (0.0, 200.0), (-0.3, 0.0), (np.log(0.1), np.log(1e5))]

    @classmethod
    def loglik_grid(cls, p, r, d, M=None):
        return grid_filter_tv(p, r, d, M or cls.GRID_M, want_pit=False)[0]

    @classmethod
    def pit_grid(cls, p, r, d, M=1600):
        return grid_filter_tv(p, r, d, M, want_pit=True)[1]

    @classmethod
    def loglik_pf(cls, p, r, d, n_particles=100_000, seed=101):
        return particle_filter_tv(p, r, d, n_particles, seed)

    @classmethod
    def fit_grid(cls, r, d, starts, M=None):
        def nll(x):
            v = -cls.loglik_grid(cls._unpack(x), r, d, M)
            return v if np.isfinite(v) else 1e12

        lo, hi = [b[0] for b in cls.BOUNDS], [b[1] for b in cls.BOUNDS]
        best = None
        for p0 in starts:
            res = optimize.minimize(nll, np.clip(cls._pack(p0), lo, hi), method="L-BFGS-B", bounds=cls.BOUNDS,
                                    options={"maxiter": 300, "eps": 1e-5})
            print(f"  BarrierBates start -> nll={res.fun:.3f}  {cls._unpack(res.x)}", flush=True)
            if best is None or res.fun < best.fun:
                best = res
        return cls._unpack(best.x), -best.fun

    @staticmethod
    def simulate(p, n, rng, warmup_prices=None, burn=500):
        # One path of n daily log returns. The barrier is rebuilt from the SIMULATED prices (rolling
        # 252-day high, kept with a monotone deque), seeded with the real 252 pre-sample closes so B is
        # defined from day one. Variance is burned in for 500 days first (no returns recorded).
        mu, kappa, theta, xi, rho, sJ = p["mu"], p["kappa"], p["theta"], p["xi"], p["rho"], p["sJ"]
        sq = np.sqrt(1 - rho ** 2)
        v = theta
        for _ in range(burn):
            v = max(v + kappa * (theta - v) * DT + xi * np.sqrt(v * DT) * rng.standard_normal(), V_FLOOR)
        hist = deque(np.log(warmup_prices[-BARRIER_WINDOW:]), maxlen=BARRIER_WINDOW)
        maxq = deque()
        for i, x in enumerate(hist):
            while maxq and maxq[-1][1] <= x:
                maxq.pop()
            maxq.append((i, x))
        idx = len(hist) - 1
        lnS = hist[-1]
        out = np.empty(n)
        e1, e2, zj, uj = rng.standard_normal(n), rng.standard_normal(n), rng.standard_normal(n), rng.random(n)
        for t in range(n):
            dd = max(0.0, lnS - (np.log(BARRIER_C) + maxq[0][1]))
            g = np.exp(-p["gamma"] * dd * dd)
            lam = p["lam0"] + p["a"] * g
            muJ = p["muJ0"] + p["b"] * g
            kbar = np.exp(muJ + 0.5 * sJ ** 2) - 1
            nj = stats.poisson.ppf(uj[t], lam * DT)
            rt = (mu - 0.5 * v - lam * kbar) * DT + np.sqrt(v * DT) * e1[t] + nj * muJ + np.sqrt(nj) * sJ * zj[t]
            out[t] = rt
            v = max(v + kappa * (theta - v) * DT + xi * np.sqrt(v * DT) * (rho * e1[t] + sq * e2[t]), V_FLOOR)
            lnS += rt
            idx += 1
            while maxq and maxq[-1][1] <= lnS:
                maxq.pop()
            maxq.append((idx, lnS))
            while maxq[0][0] <= idx - BARRIER_WINDOW:
                maxq.popleft()
        return out


def bates_base(sJ_floor):
    # Bates estimates mapped to the barrier parameterisation (lam -> lam0, muJ -> muJ0)
    pb = load_fit("bates")["params"]
    return {"mu": pb["mu"], "kappa": pb["kappa"], "theta": pb["theta"], "xi": pb["xi"], "rho": pb["rho"],
            "lam0": pb["lam"], "muJ0": pb["muJ"], "sJ": max(pb["sJ"], sJ_floor)}


# Step 1 -- screen: Bates parameters held fixed, grid-filter likelihood on 60 (a, gamma, b)
# combinations; the best 3 become starting points for the full MLE.
def screen(r, d, base):
    rows = [{"a": a, "gamma": gam, "b": b, "loglik": BarrierBates.loglik_grid(dict(base, a=a, b=b, gamma=gam), r, d)}
            for a, gam, b in itertools.product([1.0, 3.0, 10.0, 30.0], [5.0, 20.0, 100.0, 500.0, 2000.0],
                                               [0.0, -0.01, -0.03])]
    return pd.DataFrame(rows).sort_values("loglik", ascending=False)


# Barrier exposure: S&P 500 trading days by distance from the trailing 252-day high. The barrier
# binds (g_t = 1) only in March-April 2020 and parts of 2022, so a, b and gamma rest on few days.
def exposure(bf):
    dd = bf["drawdown_prev"]
    bands = [("At or below barrier (drawdown >= 20%)", dd <= -0.20),
             ("Drawdown 15-20%", (dd > -0.20) & (dd <= -0.15)),
             ("Drawdown 10-15%", (dd > -0.15) & (dd <= -0.10)),
             ("Drawdown 5-10%", (dd > -0.10) & (dd <= -0.05)),
             ("Drawdown < 5%", dd > -0.05)]
    return pd.DataFrame([[lab, int(m.sum()), f"{100 * m.mean():.1f}%"] for lab, m in bands],
                        columns=["Drawdown from trailing 252-day high (previous close)", "Days", "Share"])


# Robustness of the barrier level c (paper uses 0.8): refit B = c x trailing high from 4 starts --
# essentially Bates (a ~ 0, b ~ 0), a mid point, the c = 0.8 optimum, and a frequency-only split
# (b = 0, large a) -- and keep the best. Grid M = 150, so likelihoods compare with Bates at M = 150.
def refit_barrier_level(c, r_s, p08):
    r = r_s.to_numpy()
    bf = barrier_frame(r_s.index, c=c)
    d = bf["d"].to_numpy()
    base = bates_base(0.001)
    starts = [dict(base, a=0.01, b=-0.001, gamma=20.0), dict(base, a=1.3, b=-0.05, gamma=20.0),
              dict(p08), dict(base, a=10.0, b=0.0, gamma=50.0)]
    p, ll = BarrierBates.fit_grid(r, d, starts)
    return {"c": c, "params": p, "loglik_grid_M150": ll, "days_at_or_below_barrier": int((bf["d"] == 0).sum())}


def main():
    r_s = load_returns()
    r = r_s.to_numpy()
    n = len(r)
    bf = barrier_frame(r_s.index)
    d = bf["d"].to_numpy()
    base = bates_base(0.002)

    sc = screen(r, d, base)
    sc.to_csv(RESULTS / "barrier_screen.csv", index=False)
    print("nested Bates (a = b = 0) grid loglik:",
          round(BarrierBates.loglik_grid(dict(base, a=0.0, b=0.0, gamma=100.0), r, d), 3))
    print(sc.head(8).to_string(index=False))

    # Step 2 -- full MLE (L-BFGS-B with the bounds above) from the best 3 screen points, grid M = 150
    starts = [dict(base, a=row.a, b=row.b, gamma=row.gamma) for row in sc.head(3).itertuples()]
    p, _ = BarrierBates.fit_grid(r, d, starts)

    # Step 3 -- reported log-likelihood: particle filter, 100,000 particles, mean of 3 seeds
    # (comparable with Heston and Bates); grid M = 800 for reference
    lls = [BarrierBates.loglik_pf(p, r, d, 100_000, sd) for sd in (101, 202, 303)]
    fit = info_criteria(float(np.mean(lls)), len(BarrierBates.param_names), n)
    fit["loglik_mc_se"] = float(np.std(lls, ddof=1) / np.sqrt(len(lls)))
    fit["loglik_grid_M800"] = BarrierBates.loglik_grid(p, r, d, 800)

    # Step 4 -- PIT residuals from the grid filter at M = 1,600
    z = z_from_tails(*BarrierBates.pit_grid(p, r, d, 1600))

    # Step 5 -- 1,000 simulated paths for ADF / OU, barrier rebuilt from each simulated path
    w = load_warmup_prices()
    warm = w[w.index < r_s.index[0]].to_numpy()[-BARRIER_WINDOW:]
    sims = simulate_level_stats(BarrierBates, p, n, seed0=6_000_000, warmup_prices=warm)
    save_fit("barrier_model", p, fit, z, sims, r_s.index,
             extra={"barrier": "B = 0.8 x trailing 252-day high (fixed); one-sided kernel; yesterday's close"})

    # Table 10: barrier model against GBM and the nested Bates model
    bates = load_fit("bates")
    bates["label"] = "Bates"
    table = diagnostics("Barrier model", z, sims, fit, r, compare=bates)
    table.to_csv(RESULTS / "barrier_model_table10.csv", index=False)
    g, lam_t, muJ_t = jump_paths(p, d)
    print("Barrier model estimates (annualised, under P):", {k: round(v, 6) for k, v in p.items()})
    print(f"in sample: g_t >= {g.min():.2f}, lambda_t in [{lam_t.min():.2f}, {lam_t.max():.2f}] jumps/yr, "
          f"mu_J,t in [{100 * muJ_t.max():.1f}%, {100 * muJ_t.min():.1f}%]")
    # descriptive only: a = b = 0 sits on the boundary of the parameter space, so no chi-square LR test
    print(f"log-likelihood gain over Bates: {fit['loglik'] - bates['loglik']:.1f} for 3 extra parameters")
    print(table.to_string(index=False))

    ex = exposure(bf)
    ex.to_csv(RESULTS / "barrier_exposure.csv", index=False)
    print(ex.to_string(index=False))

    # Optional: python -m barrier.barrier_model --robustness  (refits c = 0.70 ... 0.90; several hours)
    if "--robustness" in sys.argv:
        rob = [refit_barrier_level(c, r_s, p) for c in (0.70, 0.75, 0.80, 0.85, 0.90)]
        with open(RESULTS / "barrier_level_robustness.json", "w") as f:
            json.dump(rob, f, indent=2, default=float)
        for x in rob:
            print(f"c = {x['c']:.2f}: loglik(M150) {x['loglik_grid_M150']:,.2f}, days at/below B "
                  f"{x['days_at_or_below_barrier']}, a = {x['params']['a']:.3f}, b = {x['params']['b']:.4f}, "
                  f"gamma = {x['params']['gamma']:.2f}")


if __name__ == "__main__":
    main()
