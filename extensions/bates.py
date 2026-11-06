from __future__ import annotations

import numpy as np

from common import RESULTS, diagnostics, load_fit, load_returns, save_fit, simulate_level_stats
from extensions.heston import Heston, finalize


# Bates (1996): Heston stochastic volatility plus log-normal Merton jumps in returns,
#   r_t = (mu - v_t/2 - lam kbar) dt + sqrt(v_t dt) eps1_t + sum_{i=1}^{N_t} ln J_i,
#   N_t ~ Poisson(lam dt), ln J ~ N(muJ, sJ^2), kbar = exp(muJ + sJ^2/2) - 1.
# Same Euler discretisation, grid filter and particle filter as heston.py, with the jump branch on.
class Bates(Heston):
    name = "Bates"
    param_names = ["mu", "kappa", "theta", "xi", "rho", "lam", "muJ", "sJ"]
    jumps = True

    @classmethod
    def _unpack(cls, x):
        p = super()._unpack(x[:5])
        p.update({"lam": np.exp(x[5]), "muJ": x[6], "sJ": np.exp(x[7])})
        return p

    @classmethod
    def _pack(cls, p):
        return super()._pack(p) + [np.log(p["lam"]), p["muJ"], np.log(p["sJ"])]

    @classmethod
    def _bounds_ok(cls, x):
        # ln sJ > -7 is the lower bound sJ reaches in the fit (jump size effectively fixed)
        return super()._bounds_ok(x) and (-4 < x[5] < 6) and (-0.3 < x[6] < 0.3) and (-7 < x[7] < -0.5)

    @staticmethod
    def simulate(p, n, rng, v0=None, burn=500, **_):
        return Heston.simulate(p, n, rng, v0=v0, burn=burn, jumps=True)


def main():
    r_s = load_returns()
    r = r_s.to_numpy()

    # start from the Heston optimum plus two jump configurations: a few large jumps per year
    # (~3/yr of -3%) and many small ones (~20/yr of -0.5%)
    ph = load_fit("heston")["params"]
    starts = [dict(ph, lam=3.0, muJ=-0.03, sJ=0.03), dict(ph, lam=20.0, muJ=-0.005, sJ=0.015)]
    p, _ = Bates.fit_grid(r, starts, polish_iter=600)
    fit, z = finalize(Bates, p, r)
    sims = simulate_level_stats(Bates, p, len(r), seed0=5_000_000)
    save_fit("bates", p, fit, z, sims, r_s.index)

    table = diagnostics("Bates", z, sims, fit, r)
    table.to_csv(RESULTS / "bates_table9.csv", index=False)
    print("Bates estimates (annualised, under P):", {k: round(v, 6) for k, v in p.items()})
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
