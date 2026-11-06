from __future__ import annotations

import numpy as np
from scipy import special, stats

from common import (DT, RESULTS, diagnostics, info_criteria, load_returns, multistart, save_fit,
                    simulate_level_stats, z_from_tails)


# Variance Gamma (Madan and Seneta 1990; Madan, Carr and Chang 1998): Brownian motion with drift
# run on a random gamma 'business clock',
#   r = m dt + theta G + sigma W(G),   G ~ Gamma(shape dt/nu, scale nu)
# m absorbs the drift mu and the convexity correction omega; nu is the clock's variance rate.
class VG:
    name = "VG"
    param_names = ["m", "sigma", "theta", "nu"]
    GL_NODES = 3000        # Gauss-Legendre nodes for the CDF integral over the gamma clock

    @staticmethod
    def _unpack(x):
        return {"m": x[0], "sigma": np.exp(x[1]), "theta": x[2], "nu": np.exp(x[3])}

    @staticmethod
    def logpdf(p, r):
        # Madan-Carr-Chang closed-form density of the time-dt increment (modified Bessel K),
        # with log K_v(z) = log(kve(v, z)) - z for numerical stability
        s, th, nu, t = p["sigma"], p["theta"], p["nu"], DT
        x = r - p["m"] * t
        x = np.where(np.abs(x) < 1e-14, 1e-14, x)
        a = 2 * s ** 2 / nu + th ** 2
        order = t / nu - 0.5
        arg = np.sqrt(x ** 2 * a) / s ** 2
        logK = np.log(special.kve(order, arg)) - arg
        return (np.log(2.0) + th * x / s ** 2 - (t / nu) * np.log(nu) - 0.5 * np.log(2 * np.pi) - np.log(s)
                - special.gammaln(t / nu) + (t / (2 * nu) - 0.25) * np.log(x ** 2 / a) + logK)

    @staticmethod
    def loglik(p, r):
        return float(np.sum(VG.logpdf(p, r)))

    @staticmethod
    def pit(p, r):
        # F(x) = E_G[ Phi((x - theta G) / (sigma sqrt G)) ], integrated over the gamma clock in
        # probability space (G = Gamma^{-1}(w), w in (0,1)) with Gauss-Legendre quadrature
        s, th, nu = p["sigma"], p["theta"], p["nu"]
        x = r - p["m"] * DT
        nodes, weights = np.polynomial.legendre.leggauss(VG.GL_NODES)
        w, wt = 0.5 * (nodes + 1.0), 0.5 * weights
        g = np.maximum(stats.gamma.ppf(w, a=DT / nu, scale=nu), 1e-300)
        zz = (x[:, None] - th * g[None, :]) / (s * np.sqrt(g[None, :]))
        return np.clip(stats.norm.cdf(zz) @ wt, 0.0, 1.0), np.clip(stats.norm.sf(zz) @ wt, 0.0, 1.0)

    @staticmethod
    def fit(r):
        # MLE under P: Nelder-Mead from 6 starts (symmetric and left-skewed, three clock variances)
        s, mu = r.std() / np.sqrt(DT), r.mean() / DT

        def nll(x):
            if not (-8 < x[1] < 2 and -5 < x[2] < 5 and -12 < x[3] < 3):
                return 1e12
            v = -VG.loglik(VG._unpack(x), r)
            return v if np.isfinite(v) else 1e12

        starts = [[mu, np.log(s), th, np.log(nu)] for th in (-0.2, 0.0) for nu in (0.002, 0.02, 0.2)]
        res = multistart(nll, starts)
        return VG._unpack(res.x), -res.fun

    @staticmethod
    def simulate(p, n, rng, **_):
        g = rng.gamma(shape=DT / p["nu"], scale=p["nu"], size=n)
        return p["m"] * DT + p["theta"] * g + p["sigma"] * np.sqrt(g) * rng.standard_normal(n)


def main():
    r_s = load_returns()
    r = r_s.to_numpy()
    p, ll = VG.fit(r)
    fit = info_criteria(ll, len(VG.param_names), len(r))
    z = z_from_tails(*VG.pit(p, r))
    sims = simulate_level_stats(VG, p, len(r), seed0=4_000_000)
    save_fit("variance_gamma", p, fit, z, sims, r_s.index)

    table = diagnostics("VG", z, sims, fit, r)
    table.to_csv(RESULTS / "variance_gamma_table8.csv", index=False)
    print("VG estimates (annualised, under P):", {k: round(v, 6) for k, v in p.items()})
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
