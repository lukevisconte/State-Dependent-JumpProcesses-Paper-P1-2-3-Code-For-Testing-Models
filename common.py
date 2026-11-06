from __future__ import annotations

import json
import os
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import optimize, stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller

# Sample: ^GSPC daily closes, 2016-09-01 to 2026-09-14 (yfinance end date is exclusive,
# so the last close is 2026-09-11), N = 2,519 log returns.
# The warm-up file starts 2015-08-01 so the barrier model's trailing 252-day high is
# defined from the first sample day.
HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = HERE / "results"
TICKER = "^GSPC"
START, END = "2016-09-01", "2026-09-14"
WARMUP_START = "2015-08-01"
PRICE_CSV = DATA / "sp500_prices_2016-09-01_2026-09-14.csv"
WARMUP_CSV = DATA / "sp500_prices_2015-08-01_2026-09-14_warmup.csv"

TRADING_DAYS = 252
DT = 1.0 / TRADING_DAYS
LB_LAGS = [10, 20, 100]
VOL_WINDOWS = [21, 63]
VR_Q_PRICE = [2, 5, 10, 20, 60]
VR_Q_VOL = [2, 5, 10, 21]
N_SIMS = 1000
N_NULL_VR = 2000
N_WORKERS = min(6, os.cpu_count() or 1)


# Data. Prices are read from the cached CSVs used in the paper; if a file is missing it is
# re-downloaded from yfinance (a fresh pull can differ slightly if Yahoo revises the series).
def download_prices(start: str, end: str, path: Path) -> pd.DataFrame:
    import yfinance as yf

    df = yf.download(TICKER, start=start, end=end, progress=False, auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Close"]].rename(columns={"Close": "S_t"})
    df.index.name = "Date"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    return df


def load_prices() -> pd.DataFrame:
    if not PRICE_CSV.exists():
        return download_prices(START, END, PRICE_CSV)
    return pd.read_csv(PRICE_CSV, index_col=0, parse_dates=True)


def load_warmup_prices() -> pd.Series:
    if not WARMUP_CSV.exists():
        return download_prices(WARMUP_START, END, WARMUP_CSV)["S_t"]
    return pd.read_csv(WARMUP_CSV, index_col=0, parse_dates=True)["S_t"]


def build_log_returns(prices: pd.DataFrame) -> pd.Series:
    # r_t = ln(S_t / S_{t-1})
    r = np.log(prices["S_t"]).diff().dropna()
    r.name = "r_t"
    return r


def load_returns() -> pd.Series:
    return build_log_returns(load_prices())


# Section 4 test battery.
def skewness_kurtosis(r: pd.Series) -> dict:
    # population (1/N) moments; kurtosis is raw, so the normal value is 3
    return {"skewness": stats.skew(r, bias=True),
            "kurtosis": stats.kurtosis(r, fisher=False, bias=True)}


def jarque_bera_test(r: pd.Series) -> dict:
    # JB = N/6 (S^2 + (K-3)^2 / 4), chi-square(2) under normality
    res = stats.jarque_bera(r)
    return {"jb_stat": res.statistic, "jb_pvalue": res.pvalue}


def ljung_box_tests(r: pd.Series, lags: list[int]) -> pd.DataFrame:
    # Q on raw returns (linear dependence) and on squared returns (volatility clustering)
    raw = acorr_ljungbox(r, lags=lags, return_df=True)
    raw.columns = ["Q_raw", "p_raw"]
    sq = acorr_ljungbox(r ** 2, lags=lags, return_df=True)
    sq.columns = ["Q_squared", "p_squared"]
    out = raw.join(sq)
    out.index.name = "lag"
    return out


def realized_volatility(r: pd.Series, window: int) -> pd.Series:
    # rolling annualised realized volatility
    rv = r.rolling(window).std(ddof=1) * np.sqrt(TRADING_DAYS)
    return rv.dropna().rename(f"rv_{window}")


def adf_test(series: pd.Series, label: str, alpha: float = 0.05) -> dict:
    # augmented Dickey-Fuller with a constant, lag length chosen by AIC
    series = series.dropna()
    stat, pvalue, used_lag, nobs, crit, _ = adfuller(series, autolag="AIC")
    return {"series": label, "n": nobs, "used_lag": used_lag, "adf_stat": stat, "pvalue": pvalue,
            "crit_1pct": crit["1%"], "crit_5pct": crit["5%"], "crit_10pct": crit["10%"],
            "reject_unit_root": pvalue < alpha}


def ou_ar1_hac(series: pd.Series, dt: float, maxlags: int) -> dict:
    # X_t = a + b X_{t-1} + e_t with Newey-West (HAC) standard errors.
    # Exact OU discretisation: b = exp(-kappa dt), so kappa = -ln(b)/dt, theta = a/(1-b),
    # half-life = ln 2 / kappa. Only defined when 0 < b < 1.
    series = series.dropna()
    lagged = pd.DataFrame({"y": series, "x": series.shift(1)}).dropna()
    fit = sm.OLS(lagged["y"].values, sm.add_constant(lagged["x"].values)).fit(
        cov_type="HAC", cov_kwds={"maxlags": maxlags})
    a, b = fit.params
    out = {"n_obs": int(fit.nobs), "a": a, "b": b, "b_se_hac": fit.bse[1],
           "b_pvalue_hac": fit.pvalues[1], "r_squared": fit.rsquared}
    if 0 < b < 1:
        kappa = -np.log(b) / dt
        hl = np.log(2) / kappa
        out.update(kappa_annual=kappa, theta=a / (1 - b), half_life_years=hl,
                   half_life_trading_days=hl * TRADING_DAYS, valid_mean_reverting_fit=True)
    else:
        out.update(kappa_annual=np.nan, theta=np.nan, half_life_years=np.nan,
                   half_life_trading_days=np.nan, valid_mean_reverting_fit=False)
    return out


def variance_ratio_test(level: pd.Series, q_values: list[int], label: str) -> pd.DataFrame:
    # Lo-MacKinlay (1988) variance ratio on a LEVEL series (differenced internally),
    # following Campbell, Lo and MacKinlay (1997), ch. 2.4.
    # z_homo assumes homoskedastic increments; z_hetero is the robust M2 statistic (the one quoted).
    x = level.dropna().to_numpy()
    d = np.diff(x)
    nq = len(d)
    mu = d.mean()
    dm = d - mu
    sigma_a2 = np.sum(dm ** 2) / (nq - 1)
    sq = dm ** 2
    denom = np.sum(sq) ** 2
    rows = []
    for q in q_values:
        if q < 2 or q >= nq:
            rows.append({"series": label, "q": q, "vr": np.nan, "z_homo": np.nan, "p_homo": np.nan,
                         "z_hetero": np.nan, "p_hetero": np.nan})
            continue
        dq = x[q:] - x[:-q]
        m = q * (nq - q + 1) * (1 - q / nq)
        vr = (np.sum((dq - q * mu) ** 2) / m) / sigma_a2
        z_homo = (vr - 1) / np.sqrt(2 * (2 * q - 1) * (q - 1) / (3 * q * nq))
        theta_star = sum((2 * (q - j) / q) ** 2 * nq * np.sum(sq[j:] * sq[:-j]) / denom for j in range(1, q))
        phi_star = theta_star / nq
        z_het = (vr - 1) / np.sqrt(phi_star) if phi_star > 0 else np.nan
        rows.append({"series": label, "q": q, "vr": vr,
                     "z_homo": z_homo, "p_homo": 2 * (1 - stats.norm.cdf(abs(z_homo))),
                     "z_hetero": z_het,
                     "p_hetero": 2 * (1 - stats.norm.cdf(abs(z_het))) if not np.isnan(z_het) else np.nan})
    return pd.DataFrame(rows)


def level_from_increments(x: pd.Series) -> pd.Series:
    # cumulative level whose first differences are x (the log-price analogue of a residual series)
    return pd.Series(np.concatenate([[0.0], np.cumsum(np.asarray(x, float))]))


def battery_levels(r: np.ndarray) -> dict[str, float]:
    # Level statistics for one return path: ADF on log-price, RV21, RV63 and OU/AR(1) on RV21, RV63.
    # Run on the S&P 500 and on every simulated path.
    rs = pd.Series(r)
    out = {}
    a = adf_test(level_from_increments(r), "log-price")
    out["adf_logprice"], out["adf_logprice_reject"] = a["adf_stat"], a["reject_unit_root"]
    for w in VOL_WINDOWS:
        rv = realized_volatility(rs, w)
        a = adf_test(rv, f"rv{w}")
        out[f"adf_rv{w}"], out[f"adf_rv{w}_reject"] = a["adf_stat"], a["reject_unit_root"]
        ou = ou_ar1_hac(rv, dt=DT, maxlags=w)
        out[f"ou_b_rv{w}"] = ou["b"]
        out[f"ou_kappa_rv{w}"] = ou["kappa_annual"]
        out[f"ou_theta_rv{w}"] = ou["theta"]
        out[f"ou_hl_rv{w}"] = ou["half_life_trading_days"]
    return out


# Likelihood helpers shared by the models.
def norm_logpdf(x, m, s2):
    return -0.5 * (np.log(2 * np.pi * s2) + (x - m) ** 2 / s2)


def jump_nmax(lam_dt: float) -> int:
    # truncation of the Poisson jump-count sum: P(N > nmax) < 1e-12, at least 3
    return int(max(3, stats.poisson.isf(1e-12, lam_dt) + 1))


def z_from_tails(u, s):
    # PIT residual z_t = Phi^{-1}(u_t), taken from whichever tail is smaller so extreme days
    # (e.g. 16 March 2020) keep full double precision. u = F(r_t | past), s = 1 - F(r_t | past).
    u, s = np.asarray(u, float), np.asarray(s, float)
    return np.where(u < s, stats.norm.ppf(u), stats.norm.isf(s))


def multistart(neg_ll, starts, method="Nelder-Mead", maxiter=4000):
    # Nelder-Mead from every start, keep the best, then polish with tighter tolerances
    best = None
    for x0 in starts:
        res = optimize.minimize(neg_ll, x0, method=method,
                                options={"maxiter": maxiter, "xatol": 1e-6, "fatol": 1e-6})
        if best is None or res.fun < best.fun:
            best = res
    res = optimize.minimize(neg_ll, best.x, method=method,
                            options={"maxiter": maxiter, "xatol": 1e-8, "fatol": 1e-8})
    return res if res.fun <= best.fun else best


def info_criteria(ll: float, k: int, n: int) -> dict:
    return {"loglik": ll, "k": k, "AIC": 2 * k - 2 * ll, "BIC": k * np.log(n) - 2 * ll}


# Model-simulation test for the level statistics: simulate N_SIMS paths of N returns from the
# fitted model and compute battery_levels on each. Seeds are fixed so the paper's numbers repeat.
def _sim_one(seed, model, params, n, kwargs):
    rng = np.random.default_rng(seed)
    return battery_levels(model.simulate(params, n, rng, **kwargs))


def simulate_level_stats(model, params, n, seed0, sims=N_SIMS, **kwargs) -> pd.DataFrame:
    f = partial(_sim_one, model=model, params=params, n=n, kwargs=kwargs)
    with Pool(N_WORKERS) as pool:
        rows = pool.map(f, range(seed0, seed0 + sims), chunksize=20)
    return pd.DataFrame(rows)


# Monte Carlo null for the variance ratio on realized vol of PIT residuals. The asymptotic
# Lo-MacKinlay p-values are invalid for an overlapping rolling-window series; under a correct
# model z_t is i.i.d. N(0,1), so 2,000 i.i.d. N(0,1) series of length N give the right null for
# every model at once (overlap distortion included). Cached in results/.
def _null_one(seed, n):
    z = pd.Series(np.random.default_rng(seed).standard_normal(n))
    row = {}
    for w in VOL_WINDOWS:
        vr = variance_ratio_test(realized_volatility(z, w), [q for q in VR_Q_VOL if q < w], "")
        for _, x in vr.iterrows():
            row[f"vr_rv{w}_q{int(x.q)}"] = x.vr
    return row


def mc_null_vr(n: int = 2519, reps: int = N_NULL_VR) -> pd.DataFrame:
    path = RESULTS / "mc_null_vr.csv"
    if path.exists():
        return pd.read_csv(path)
    with Pool(N_WORKERS) as pool:
        rows = pool.map(partial(_null_one, n=n), range(10_000, 10_000 + reps), chunksize=50)
    RESULTS.mkdir(exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return df


def mc_p(sims, x) -> float:
    # two-sided Monte Carlo p-value: 2 min(P(sim <= x), P(sim >= x))
    sims = np.asarray(sims, float)
    sims = sims[~np.isnan(sims)]
    return float(min(1.0, 2 * min(np.mean(sims <= x), np.mean(sims >= x))))


# Saving and loading fitted models so later scripts can build on earlier ones
# (Bates starts from Heston, the barrier model from Bates, every table compares with GBM).
def save_fit(tag: str, params: dict, fit: dict, z: np.ndarray, sims: pd.DataFrame, index, extra=None):
    RESULTS.mkdir(exist_ok=True)
    out = {"params": params, **fit, **(extra or {})}
    with open(RESULTS / f"{tag}_params.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    pd.DataFrame({"z": z}, index=index).to_csv(RESULTS / f"{tag}_pit.csv")
    sims.to_csv(RESULTS / f"{tag}_sims.csv", index=False)


def load_fit(tag: str) -> dict:
    path = RESULTS / f"{tag}_params.json"
    if not path.exists():
        raise FileNotFoundError(f"{path.name} not found -- run that model's script first (see paper_results.md for the order)")
    with open(path) as f:
        out = json.load(f)
    out["z"] = pd.read_csv(RESULTS / f"{tag}_pit.csv", index_col=0)["z"].to_numpy()
    out["sims"] = pd.read_csv(RESULTS / f"{tag}_sims.csv")
    return out


# Paper Tables 6-10: the Section 4 statistics on a model's PIT residuals (Panels A-C), the
# simulated level statistics against the S&P 500 (Panel D) and the fit (Panel E), next to GBM.
def _pit_stats(z) -> dict:
    x = pd.Series(np.asarray(z, float)).dropna().reset_index(drop=True)
    sk, jb, lb = skewness_kurtosis(x), jarque_bera_test(x), ljung_box_tests(x, LB_LAGS)
    lvl = variance_ratio_test(level_from_increments(x), [2, 20], "level").set_index("q")
    rv21 = variance_ratio_test(realized_volatility(x, 21), [10], "rv21").iloc[0]
    rv63 = variance_ratio_test(realized_volatility(x, 63), [21], "rv63").iloc[0]
    return {"skew": sk["skewness"], "kurt": sk["kurtosis"], "jb": jb["jb_stat"], "jb_p": jb["jb_pvalue"],
            "qr10": lb.loc[10, "Q_raw"], "qr10_p": lb.loc[10, "p_raw"],
            "qr100": lb.loc[100, "Q_raw"], "qr100_p": lb.loc[100, "p_raw"],
            "qs10": lb.loc[10, "Q_squared"], "qs10_p": lb.loc[10, "p_squared"],
            "qs100": lb.loc[100, "Q_squared"], "qs100_p": lb.loc[100, "p_squared"],
            "vr2": lvl.loc[2, "vr"], "vr2_p": lvl.loc[2, "p_hetero"],
            "vr20": lvl.loc[20, "vr"], "vr20_p": lvl.loc[20, "p_hetero"],
            "rv21": rv21.vr, "rv63": rv63.vr}


def _fp(p) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def _band(s, d) -> str:
    s = np.asarray(s, float)
    return f"{np.nanmedian(s):.{d}f} [{np.nanquantile(s, .05):.{d}f}, {np.nanquantile(s, .95):.{d}f}]"


def diagnostics(label: str, z, sims: pd.DataFrame, fit: dict, r: np.ndarray, compare: dict | None = None) -> pd.DataFrame:
    gbm = load_fit("gbm")
    null = mc_null_vr(len(r))
    data_lv = battery_levels(r)
    g = _pit_stats(r)                       # GBM's PIT residual is the standardised return
    s = _pit_stats(z)
    c = _pit_stats(compare["z"]) if compare else None
    cols = ["Statistic", "GBM (Part 4)"] + ([compare["label"]] if compare else []) + [label, "Reference", f"p-value ({label})"]
    rows = []

    def add(stat, gv, cv, mv, ref, p):
        rows.append([stat, gv] + ([cv] if compare else []) + [mv, ref, p])

    def panel(title):
        rows.append([title] + [""] * (len(cols) - 1))

    panel("Panel A: Distribution of PIT residuals")
    add("Skewness", f"{g['skew']:.3f}", c and f"{c['skew']:.3f}", f"{s['skew']:.3f}", "0", "")
    add("Kurtosis", f"{g['kurt']:.2f}", c and f"{c['kurt']:.3f}", f"{s['kurt']:.3f}", "3", "")
    add("Jarque-Bera", f"{g['jb']:,.1f}", c and f"{c['jb']:,.1f}", f"{s['jb']:,.1f}", "chi2(2)", _fp(s["jb_p"]))

    panel("Panel B: Serial dependence (Ljung-Box)")
    for key, stat, ref in [("qr10", "Q_raw(10)", "chi2(10)"), ("qr100", "Q_raw(100)", "chi2(100)"),
                           ("qs10", "Q_sq(10)", "chi2(10)"), ("qs100", "Q_sq(100)", "chi2(100)")]:
        add(stat, f"{g[key]:,.1f}", c and f"{c[key]:,.1f}", f"{s[key]:,.1f}", ref, _fp(s[key + "_p"]))

    panel("Panel C: Variance ratios")
    add("VR(2), cumulated returns / z", f"{g['vr2']:.3f}", c and f"{c['vr2']:.3f}", f"{s['vr2']:.3f}", "1", _fp(s["vr2_p"]))
    add("VR(20), cumulated returns / z", f"{g['vr20']:.3f}", c and f"{c['vr20']:.3f}", f"{s['vr20']:.3f}", "1", _fp(s["vr20_p"]))
    add("VR(10), realized vol 21d", f"{g['rv21']:.3f}", c and f"{c['rv21']:.3f}", f"{s['rv21']:.3f}", "1",
        _fp(mc_p(null["vr_rv21_q10"], s["rv21"])) + " (MC)")
    add("VR(21), realized vol 63d", f"{g['rv63']:.3f}", c and f"{c['rv63']:.3f}", f"{s['rv63']:.3f}", "1",
        _fp(mc_p(null["vr_rv63_q21"], s["rv63"])) + " (MC)")

    # GBM and comparison columns: simulated medians; model column: median [5%, 95%] band.
    # OU p-values are computed on the AR(1) coefficient b (monotone in the half-life, defined on every path).
    panel(f"Panel D: Level statistics, model-implied vs. S&P 500 ({len(sims):,} simulated paths)")
    gs, cs = gbm["sims"], compare["sims"] if compare else None
    for col, stat, d in [("adf_logprice", "ADF, log price", 2), ("adf_rv21", "ADF, realized vol 21d", 2),
                         ("ou_hl_rv21", "OU half-life, RV 21d (days)", 1), ("ou_hl_rv63", "OU half-life, RV 63d (days)", 1)]:
        p = mc_p(sims[f"ou_b_rv{col[-2:]}"], data_lv[f"ou_b_rv{col[-2:]}"]) if col.startswith("ou") else mc_p(sims[col], data_lv[col])
        add(stat, f"{np.nanmedian(gs[col]):.{d}f}", compare and f"{np.nanmedian(cs[col]):.{d}f}",
            _band(sims[col], d), f"{data_lv[col]:.{d}f} (S&P)", _fp(p) + " (MC)")

    panel("Panel E: Fit")
    add("Log-likelihood", f"{gbm['loglik']:,.1f}", compare and f"{compare['loglik']:,.1f}", f"{fit['loglik']:,.1f}", "", "")
    add("AIC", f"{gbm['AIC']:,.1f}", compare and f"{compare['AIC']:,.1f}", f"{fit['AIC']:,.1f}", "lower is better", "")
    add("BIC", f"{gbm['BIC']:,.1f}", compare and f"{compare['BIC']:,.1f}", f"{fit['BIC']:,.1f}", "lower is better", "")
    add("Parameters (k)", f"{gbm['k']}", compare and f"{compare['k']}", f"{fit['k']}", "", "")
    return pd.DataFrame(rows, columns=cols)
