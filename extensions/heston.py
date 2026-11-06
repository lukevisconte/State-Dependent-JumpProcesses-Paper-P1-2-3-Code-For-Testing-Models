from __future__ import annotations

import numpy as np
from scipy import optimize, special, stats

from common import (DT, RESULTS, diagnostics, info_criteria, jump_nmax, load_returns, norm_logpdf, save_fit,
                    simulate_level_stats, z_from_tails)

V_FLOOR = 1e-6                    # variance floor (annualised; = 0.1% vol), shared by filters and simulator
VOL_LO, VOL_HI = 0.001, 1.60      # grid range in volatility; VOL_LO^2 = V_FLOOR


# Heston (1993) under P, daily Euler discretisation (the same equations are used for estimation,
# PIT and simulation, so all three refer to one model):
#   r_t     = (mu - v_t/2 - lam kbar) dt + sqrt(v_t dt) eps1_t + J_t
#   v_{t+1} = v_t + kappa (theta - v_t) dt + xi sqrt(v_t dt) (rho eps1_t + sqrt(1 - rho^2) eta_t)
# J_t is a sum of N_t log-normal jumps, N_t ~ Poisson(lam dt). Heston has lam = 0; bates.py turns
# the jump branch on. The variance v_t is latent, so the likelihood needs a filter.


# Deterministic grid (HMM) filter, used for estimation and for the PIT (Kitagawa 1987).
# v is discretised on M cells uniform in volatility between VOL_LO and VOL_HI. With
# pi_t(i) = P(v_t in cell i | r_1..r_{t-1}):
#   likelihood_t = sum_i pi_t(i) f(r_t | v_i),   PIT u_t = sum_i pi_t(i) F(r_t | v_i),
#   pi_{t+1}(j)  = sum_i post_t(i) K_t(i, j),
# where K_t(i, j) is the exact probability that the Euler step moves v from cell i to cell j given
# r_t (the return pins down the correlated shock eps1; with jumps, mix over the jump count).
# The likelihood is smooth in the parameters, so gradient-based MLE is reliable.
def grid_filter(p, r, M, jumps: bool, want_pit: bool):
    mu, kappa, theta, xi, rho = p["mu"], p["kappa"], p["theta"], p["xi"], p["rho"]
    lam = p.get("lam", 0.0) if jumps else 0.0
    muJ, sJ = p.get("muJ", 0.0), p.get("sJ", 1e-3)
    kbar = np.exp(muJ + 0.5 * sJ ** 2) - 1 if jumps else 0.0

    edges = np.linspace(VOL_LO, VOL_HI, M + 1)
    vc = (0.5 * (edges[1:] + edges[:-1])) ** 2          # cell centres (variance)
    B = edges[1:-1] ** 2                                 # interior cell boundaries (variance)
    vd = vc * DT
    sdd = np.sqrt(vd)
    m = (mu - 0.5 * vc - lam * kbar) * DT
    drift_v = vc + kappa * (theta - vc) * DT

    # initial distribution: stationary CIR law Gamma(2 kappa theta / xi^2, xi^2 / (2 kappa)) on the cells
    cdf = stats.gamma.cdf(edges ** 2, 2 * kappa * theta / xi ** 2, scale=xi ** 2 / (2 * kappa))
    cdf[0], cdf[-1] = 0.0, 1.0
    pi = np.maximum(np.diff(cdf), 0.0)
    pi /= pi.sum()

    if jumps:
        n_grid = np.arange(jump_nmax(lam * DT) + 1)
        logpn = stats.poisson.logpmf(n_grid, lam * DT)
        pn = np.exp(logpn)
        mn = m[:, None] + n_grid[None, :] * muJ
        vn = vd[:, None] + n_grid[None, :] * sJ ** 2
        jv = n_grid[None, :] * sJ ** 2
        gain = vd[:, None] / (vd[:, None] + jv)
        ratio_var = jv / (vd[:, None] + jv)              # Var(eps1 | r, n)

    T = len(r)
    loglik = 0.0
    u_out = np.empty(T) if want_pit else None
    s_out = np.empty(T) if want_pit else None
    sq2 = 1 - rho ** 2
    for t in range(T):
        rt = r[t]
        if not jumps:
            logf = norm_logpdf(rt, m, vd)
            if want_pit:
                xx = (rt - m) / sdd
                u_out[t], s_out[t] = pi @ special.ndtr(xx), pi @ special.ndtr(-xx)
        else:
            comp = logpn[None, :] + norm_logpdf(rt, mn, vn)
            cm = comp.max(axis=1, keepdims=True)
            ec = np.exp(comp - cm)
            logf = cm[:, 0] + np.log(ec.sum(axis=1))
            wn = ec / ec.sum(axis=1, keepdims=True)                # P(n | v_i, r_t)
            if want_pit:
                xx = (rt - mn) / np.sqrt(vn)
                u_out[t], s_out[t] = pi @ (special.ndtr(xx) @ pn), pi @ (special.ndtr(-xx) @ pn)
        cmax = logf.max()
        f = np.exp(logf - cmax)
        lt = pi @ f
        loglik += cmax + np.log(lt)
        post = pi * f / lt

        # propagate: Gaussian transition of v given r_t, integrated over the target cells
        new = np.zeros(M)
        if not jumps:
            eps1 = (rt - m) / sdd
            mean_v = drift_v + xi * sdd * rho * eps1
            sd_v = xi * sdd * np.sqrt(sq2)
            keep = post > 1e-14
            C = special.ndtr((B[None, :] - mean_v[keep, None]) / sd_v[keep, None])
            K = np.diff(np.concatenate([np.zeros((C.shape[0], 1)), C, np.ones((C.shape[0], 1))], axis=1), axis=1)
            new = post[keep] @ K
        else:
            for k in range(len(n_grid)):
                q = post * wn[:, k]
                keep = q > 1e-14
                if not keep.any():
                    continue
                e_mean = gain[keep, k] * (rt - mn[keep, k]) / sdd[keep]
                mean_v = drift_v[keep] + xi * sdd[keep] * rho * e_mean
                sd_v = xi * sdd[keep] * np.sqrt(sq2 + rho ** 2 * ratio_var[keep, k])
                C = special.ndtr((B[None, :] - mean_v[:, None]) / sd_v[:, None])
                K = np.diff(np.concatenate([np.zeros((C.shape[0], 1)), C, np.ones((C.shape[0], 1))], axis=1), axis=1)
                new += q[keep] @ K
        pi = new / new.sum()
    return loglik, (u_out, s_out)


# Particle filter: the likelihood of exactly the floored Euler model the simulator runs, used for
# the reported log-likelihoods (100,000 particles, 3 seeds) and as a cross-check on the grid PIT.
# Equally weighted particles before r_t give the one-step predictive distribution. After systematic
# resampling (state sorted first, common random numbers), each particle's eps1 is recovered from
# r_t (with jumps: after drawing the jump count and diffusive part given r_t) and v is propagated
# with the correct correlation rho.
def particle_filter(p, r, n_particles, seed, jumps: bool, want_pit: bool):
    mu, kappa, theta, xi, rho = p["mu"], p["kappa"], p["theta"], p["xi"], p["rho"]
    lam = p.get("lam", 0.0) if jumps else 0.0
    muJ, sJ = p.get("muJ", 0.0), p.get("sJ", 1e-3)
    kbar = np.exp(muJ + 0.5 * sJ ** 2) - 1 if jumps else 0.0

    T = len(r)
    rng = np.random.default_rng(seed)
    v = np.maximum(rng.gamma(2 * kappa * theta / xi ** 2, xi ** 2 / (2 * kappa), n_particles), V_FLOOR)
    U_res = rng.random(T)
    ETA = rng.standard_normal((T, n_particles))
    if jumps:
        U_n = rng.random((T, n_particles))
        Z_d = rng.standard_normal((T, n_particles))
        nmax = jump_nmax(lam * DT)
        n_grid = np.arange(nmax + 1)
        logpn = stats.poisson.logpmf(n_grid, lam * DT)
    sq = np.sqrt(1 - rho ** 2)
    loglik = 0.0
    u_out = np.empty(T) if want_pit else None
    s_out = np.empty(T) if want_pit else None
    idx_base = np.arange(n_particles)

    for t in range(T):
        rt = r[t]
        m = (mu - 0.5 * v - lam * kbar) * DT
        vd = v * DT
        if not jumps:
            logf = norm_logpdf(rt, m, vd)
            if want_pit:
                xx = (rt - m) / np.sqrt(vd)
                u_out[t], s_out[t] = np.mean(stats.norm.cdf(xx)), np.mean(stats.norm.sf(xx))
        else:
            mn = m[:, None] + n_grid[None, :] * muJ
            vn = vd[:, None] + n_grid[None, :] * sJ ** 2
            comp = logpn[None, :] + norm_logpdf(rt, mn, vn)
            cm = comp.max(axis=1, keepdims=True)
            logf = cm[:, 0] + np.log(np.exp(comp - cm).sum(axis=1))
            if want_pit:
                xx = (rt - mn) / np.sqrt(vn)
                pn = np.exp(logpn)[None, :]
                u_out[t] = np.mean(np.sum(pn * stats.norm.cdf(xx), axis=1))
                s_out[t] = np.mean(np.sum(pn * stats.norm.sf(xx), axis=1))
        cmax = logf.max()
        w = np.exp(logf - cmax)
        wsum = w.sum()
        loglik += cmax + np.log(wsum / n_particles)

        order = np.argsort(v)
        v_s, w_s = v[order], w[order] / wsum
        cw = np.cumsum(w_s)
        cw[-1] = 1.0
        pick = np.searchsorted(cw, (U_res[t] + idx_base) / n_particles)
        v_r = v_s[pick]
        m_r = (mu - 0.5 * v_r - lam * kbar) * DT
        vd_r = v_r * DT

        if not jumps:
            eps1 = (rt - m_r) / np.sqrt(vd_r)
        else:
            comp_r = comp[order][pick]
            post = np.exp(comp_r - comp_r.max(axis=1, keepdims=True))
            post /= post.sum(axis=1, keepdims=True)
            ncount = np.minimum((np.cumsum(post, axis=1) < U_n[t][:, None]).sum(axis=1), nmax)
            jvar = ncount * sJ ** 2
            gain = vd_r / (vd_r + jvar)
            d_mean = m_r + gain * (rt - m_r - ncount * muJ)
            d_var = vd_r * jvar / (vd_r + jvar)
            eps1 = (d_mean + np.sqrt(d_var) * Z_d[t] - m_r) / np.sqrt(vd_r)

        v = np.maximum(v_r + kappa * (theta - v_r) * DT + xi * np.sqrt(vd_r) * (rho * eps1 + sq * ETA[t]), V_FLOOR)

    return loglik, (u_out, s_out)


class Heston:
    name = "Heston"
    param_names = ["mu", "kappa", "theta", "xi", "rho"]
    jumps = False
    GRID_M = 150          # grid cells during the likelihood search

    # optimiser coordinates: mu, ln kappa, ln theta, ln xi, atanh rho
    @classmethod
    def _unpack(cls, x):
        return {"mu": x[0], "kappa": np.exp(x[1]), "theta": np.exp(x[2]), "xi": np.exp(x[3]), "rho": np.tanh(x[4])}

    @classmethod
    def _pack(cls, p):
        return [p["mu"], np.log(p["kappa"]), np.log(p["theta"]), np.log(p["xi"]), np.arctanh(p["rho"])]

    @classmethod
    def _bounds_ok(cls, x):
        return (-3 < x[1] < 4.5) and (-7 < x[2] < 0) and (-5 < x[3] < 2) and (-3 < x[4] < 3)

    @classmethod
    def loglik_grid(cls, p, r, M=None):
        return grid_filter(p, r, M or cls.GRID_M, cls.jumps, want_pit=False)[0]

    @classmethod
    def pit_grid(cls, p, r, M=1600):
        return grid_filter(p, r, M, cls.jumps, want_pit=True)[1]

    @classmethod
    def loglik_pf(cls, p, r, n_particles=100_000, seed=101):
        return particle_filter(p, r, n_particles, seed, cls.jumps, want_pit=False)[0]

    @classmethod
    def pit_pf(cls, p, r, n_particles=100_000, seed=2024):
        return particle_filter(p, r, n_particles, seed, cls.jumps, want_pit=True)[1]

    @classmethod
    def fit_grid(cls, r, starts, M=None, polish_iter=2000):
        # MLE with the grid filter: L-BFGS-B from each start, then a Nelder-Mead polish of the best
        def nll(x):
            if not cls._bounds_ok(x):
                return 1e12
            v = -cls.loglik_grid(cls._unpack(x), r, M)
            return v if np.isfinite(v) else 1e12

        best = None
        for p0 in starts:
            res = optimize.minimize(nll, cls._pack(p0), method="L-BFGS-B", options={"maxiter": 300, "eps": 1e-5})
            print(f"  {cls.name} start -> nll={res.fun:.3f}  {cls._unpack(res.x)}", flush=True)
            if best is None or res.fun < best.fun:
                best = res
        pol = optimize.minimize(nll, best.x, method="Nelder-Mead",
                                options={"maxiter": polish_iter, "xatol": 1e-6, "fatol": 1e-6, "adaptive": True})
        if pol.fun < best.fun:
            best = pol
        return cls._unpack(best.x), -best.fun

    @staticmethod
    def simulate(p, n, rng, v0=None, burn=500, jumps=False):
        # Euler path with the same floor as the filters; 500 burn-in days so v starts near stationarity
        mu, kappa, theta, xi, rho = p["mu"], p["kappa"], p["theta"], p["xi"], p["rho"]
        lam = p.get("lam", 0.0) if jumps else 0.0
        muJ, sJ = p.get("muJ", 0.0), p.get("sJ", 0.0)
        kbar = np.exp(muJ + 0.5 * sJ ** 2) - 1 if jumps else 0.0
        v = theta if v0 is None else v0
        out = np.empty(n)
        sq = np.sqrt(1 - rho ** 2)
        e1, e2 = rng.standard_normal(n + burn), rng.standard_normal(n + burn)
        if jumps:
            N = rng.poisson(lam * DT, n + burn)
            J = N * muJ + np.sqrt(N) * sJ * rng.standard_normal(n + burn)
        for t in range(n + burn):
            rt = (mu - 0.5 * v - lam * kbar) * DT + np.sqrt(v * DT) * e1[t]
            if jumps:
                rt += J[t]
            if t >= burn:
                out[t - burn] = rt
            v = max(v + kappa * (theta - v) * DT + xi * np.sqrt(v * DT) * (rho * e1[t] + sq * e2[t]), V_FLOOR)
        return out


# Final estimation pipeline shared with Bates:
#   1. MLE with the grid filter (M = 150)
#   2. reported log-likelihood: particle filter, 100,000 particles, mean of 3 seeds (MC s.e. kept);
#      grid M = 800 likelihood kept for reference
#   3. PIT from the grid filter at M = 1,600 (converged: M = 800 -> 1,600 moves Q_sq(10) ~1%, the
#      rest < 0.2%), with a 100,000-particle PIT as a cross-check
def finalize(model, p, r, ll_seeds=(101, 202, 303)):
    lls = [model.loglik_pf(p, r, 100_000, sd) for sd in ll_seeds]
    fit = info_criteria(float(np.mean(lls)), len(model.param_names), len(r))
    fit["loglik_mc_se"] = float(np.std(lls, ddof=1) / np.sqrt(len(lls)))
    fit["loglik_grid_M800"] = model.loglik_grid(p, r, 800)
    z = z_from_tails(*model.pit_grid(p, r, 1600))
    z_pf = z_from_tails(*model.pit_pf(p, r))
    fit["pit_grid_vs_pf100k_median_abs_dz"] = float(np.median(np.abs(z - z_pf)))
    fit["pit_grid_vs_pf100k_max_abs_dz"] = float(np.max(np.abs(z - z_pf)))
    return fit, z


def main():
    r_s = load_returns()
    r = r_s.to_numpy()
    s = r.std() / np.sqrt(DT)
    starts = [{"mu": 0.10, "kappa": 3.0, "theta": s ** 2, "xi": 0.5, "rho": -0.7},
              {"mu": 0.10, "kappa": 8.0, "theta": s ** 2, "xi": 1.0, "rho": -0.5},
              {"mu": 0.10, "kappa": 1.5, "theta": 1.3 * s ** 2, "xi": 0.3, "rho": -0.8}]
    p, _ = Heston.fit_grid(r, starts)
    fit, z = finalize(Heston, p, r)
    sims = simulate_level_stats(Heston, p, len(r), seed0=3_000_000)
    save_fit("heston", p, fit, z, sims, r_s.index)

    table = diagnostics("Heston", z, sims, fit, r)
    table.to_csv(RESULTS / "heston_table7.csv", index=False)
    print("Heston estimates (annualised, under P):", {k: round(v, 6) for k, v in p.items()})
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
