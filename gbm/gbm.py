from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from common import (DT, LB_LAGS, RESULTS, VOL_WINDOWS, VR_Q_PRICE, VR_Q_VOL, adf_test, info_criteria,
                    jarque_bera_test, ljung_box_tests, load_prices, load_returns, norm_logpdf, ou_ar1_hac,
                    realized_volatility, save_fit, simulate_level_stats, skewness_kurtosis, variance_ratio_test,
                    z_from_tails)


# GBM: dS = mu S dt + sigma S dW, so daily log returns are i.i.d.
#   r_t ~ N((mu - sigma^2/2) dt, sigma^2 dt),   dt = 1/252, mu and sigma annualised.
class GBM:
    name = "GBM"
    param_names = ["mu", "sigma"]

    @staticmethod
    def _ms(p):
        return (p["mu"] - 0.5 * p["sigma"] ** 2) * DT, p["sigma"] ** 2 * DT

    @staticmethod
    def fit(r):
        # closed-form MLE of the normal, mapped back to (mu, sigma)
        m, s2 = r.mean(), r.var(ddof=0)
        sigma = np.sqrt(s2 / DT)
        p = {"mu": m / DT + 0.5 * sigma ** 2, "sigma": sigma}
        return p, GBM.loglik(p, r)

    @staticmethod
    def loglik(p, r):
        m, s2 = GBM._ms(p)
        return float(np.sum(norm_logpdf(r, m, s2)))

    @staticmethod
    def pit(p, r):
        # u_t = Phi((r_t - m) / s): GBM's PIT residual is just the standardised return,
        # so every Section 4 statistic on z_t equals the statistic on raw returns
        m, s2 = GBM._ms(p)
        x = (r - m) / np.sqrt(s2)
        return stats.norm.cdf(x), stats.norm.sf(x)

    @staticmethod
    def simulate(p, n, rng, **_):
        m, s2 = GBM._ms(p)
        return m + np.sqrt(s2) * rng.standard_normal(n)


# Section 4: the four GBM/Gaussian diagnostics (skewness, kurtosis, Jarque-Bera, Ljung-Box) plus
# the mean-reversion tests (variance ratio, ADF, OU/AR(1)) on the log-price and on 21- and 63-day
# realized volatility. Produces paper Tables 1-5.
def section4_tables(prices: pd.DataFrame, r: pd.Series) -> dict[str, pd.DataFrame]:
    log_price = np.log(prices["S_t"]).dropna()
    n = len(r)
    sk, jb = skewness_kurtosis(r), jarque_bera_test(r)
    t1 = pd.DataFrame([{"n": n, "skewness": sk["skewness"], "kurtosis": sk["kurtosis"],
                        "normal_skewness": 0.0, "normal_kurtosis": 3.0,
                        "jb_stat": jb["jb_stat"], "jb_pvalue": jb["jb_pvalue"]}])

    t2 = ljung_box_tests(r, LB_LAGS).reset_index()

    rv = {w: realized_volatility(r, w) for w in VOL_WINDOWS}
    # VR on realized vol uses q < window; these rows are from an overlapping rolling series,
    # so their asymptotic p-values are only directional (see the Monte Carlo null in common.py)
    t3 = pd.concat([variance_ratio_test(log_price, VR_Q_PRICE, "log-price")] +
                   [variance_ratio_test(rv[w], [q for q in VR_Q_VOL if q < w], f"realized_vol_{w}d")
                    for w in VOL_WINDOWS], ignore_index=True)

    # ADF: log-price should fail to reject a unit root (random walk); rolling realized vol
    # rejects it even under constant volatility, which is why Section 6 checks it by simulation
    t4 = pd.DataFrame([adf_test(log_price, "log-price")] +
                      [adf_test(rv[w], f"realized_vol_{w}d") for w in VOL_WINDOWS])

    # OU/AR(1) on realized vol: speed kappa, long-run level theta, half-life in trading days
    t5 = pd.DataFrame([{"window": w, **ou_ar1_hac(rv[w], dt=DT, maxlags=w)} for w in VOL_WINDOWS])
    return {"table1_moments_jb": t1, "table2_ljung_box": t2, "table3_variance_ratio": t3,
            "table4_adf": t4, "table5_ou_ar1": t5}


def main():
    prices = load_prices()
    r_s = load_returns()
    r = r_s.to_numpy()
    n = len(r)

    tables = section4_tables(prices, r_s)
    RESULTS.mkdir(exist_ok=True)
    for name, df in tables.items():
        df.to_csv(RESULTS / f"gbm_{name}.csv", index=False)
        print(f"\n{name}\n{df.to_string(index=False)}")

    # GBM fit under P, its PIT residuals and 1,000 simulated paths. The simulated level
    # statistics are the 'GBM (Part 4)' column of Panel D in Tables 6-10: what overlapping
    # windows alone produce when volatility is constant (half-lives of ~13 and ~41 days).
    p, ll = GBM.fit(r)
    fit = info_criteria(ll, len(GBM.param_names), n)
    z = z_from_tails(*GBM.pit(p, r))
    sims = simulate_level_stats(GBM, p, n, seed0=1_000_000)
    save_fit("gbm", p, fit, z, sims, r_s.index)
    print("\nGBM estimates (annualised, under P):", {k: round(v, 6) for k, v in p.items()})
    print(f"log-likelihood {ll:,.1f} | AIC {fit['AIC']:,.1f} | BIC {fit['BIC']:,.1f}")
    print("GBM simulated medians:", sims.median(numeric_only=True).round(3).to_dict())


if __name__ == "__main__":
    main()
