from __future__ import annotations

import numpy as np
from scipy import special, stats

from common import (DT, RESULTS, diagnostics, info_criteria, jump_nmax, load_returns, multistart, norm_logpdf,
                    save_fit, simulate_level_stats, z_from_tails)


# Merton (1976) jump diffusion, daily log returns:
#   r = (alpha - sigma^2/2 - lam k) dt + sigma sqrt(dt) eps + sum_{i=1}^{N} ln Y_i
#   N ~ Poisson(lam dt),  ln Y ~ N(muJ, sJ^2),  k = E[Y - 1] = exp(muJ + sJ^2/2) - 1
# The density is a Poisson-weighted mixture of normals (one component per jump count),
# truncated where the remaining Poisson mass is below 1e-12.
class Merton:
    name = "Merton"
    param_names = ["alpha", "sigma", "lam", "muJ", "sJ"]

    @staticmethod
    def _unpack(x):
        # optimiser works on log sigma, log lam, log sJ so they stay positive
        return {"alpha": x[0], "sigma": np.exp(x[1]), "lam": np.exp(x[2]), "muJ": x[3], "sJ": np.exp(x[4])}

    @staticmethod
    def _components(p):
        # log mixture weights, means and variances of the n = 0..nmax jump components
        k = np.exp(p["muJ"] + 0.5 * p["sJ"] ** 2) - 1
        lam_dt = p["lam"] * DT
        n = np.arange(jump_nmax(lam_dt) + 1)
        logw = stats.poisson.logpmf(n, lam_dt)
        m0 = (p["alpha"] - 0.5 * p["sigma"] ** 2 - p["lam"] * k) * DT
        return logw, m0 + n * p["muJ"], p["sigma"] ** 2 * DT + n * p["sJ"] ** 2

    @staticmethod
    def loglik(p, r):
        logw, means, vars_ = Merton._components(p)
        comp = logw[None, :] + norm_logpdf(r[:, None], means[None, :], vars_[None, :])
        return float(np.sum(special.logsumexp(comp, axis=1)))

    @staticmethod
    def pit(p, r):
        # mixture CDF and survival function, returned separately for z_from_tails
        logw, means, vars_ = Merton._components(p)
        x = (r[:, None] - means[None, :]) / np.sqrt(vars_[None, :])
        w = np.exp(logw)[None, :]
        return np.sum(w * stats.norm.cdf(x), axis=1), np.sum(w * stats.norm.sf(x), axis=1)

    @staticmethod
    def fit(r):
        # MLE under P: Nelder-Mead from 16 starts (jump rate 2-150/yr, small and larger jumps)
        s, mu = r.std() / np.sqrt(DT), r.mean() / DT

        def nll(x):
            if not (-12 < x[1] < 2 and -6 < x[2] < 8 and -9 < x[4] < 0):
                return 1e12
            v = -Merton.loglik(Merton._unpack(x), r)
            return v if np.isfinite(v) else 1e12

        starts = [[mu, np.log(0.8 * s), np.log(lam), muJ, np.log(sJ)]
                  for lam in (2.0, 10.0, 40.0, 150.0) for sJ in (0.01, 0.03) for muJ in (-0.02, -0.002)]
        res = multistart(nll, starts)
        return Merton._unpack(res.x), -res.fun

    @staticmethod
    def simulate(p, n, rng, **_):
        k = np.exp(p["muJ"] + 0.5 * p["sJ"] ** 2) - 1
        m0 = (p["alpha"] - 0.5 * p["sigma"] ** 2 - p["lam"] * k) * DT
        N = rng.poisson(p["lam"] * DT, n)
        jumps = N * p["muJ"] + np.sqrt(N) * p["sJ"] * rng.standard_normal(n)
        return m0 + p["sigma"] * np.sqrt(DT) * rng.standard_normal(n) + jumps


def main():
    r_s = load_returns()
    r = r_s.to_numpy()
    p, ll = Merton.fit(r)
    fit = info_criteria(ll, len(Merton.param_names), len(r))
    z = z_from_tails(*Merton.pit(p, r))
    sims = simulate_level_stats(Merton, p, len(r), seed0=2_000_000)
    save_fit("merton", p, fit, z, sims, r_s.index)

    table = diagnostics("Merton", z, sims, fit, r)
    table.to_csv(RESULTS / "merton_table6.csv", index=False)
    print("Merton estimates (annualised, under P):", {k: round(v, 6) for k, v in p.items()})
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
