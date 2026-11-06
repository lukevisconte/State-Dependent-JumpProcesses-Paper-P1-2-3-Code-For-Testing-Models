Reference copy of the result tables in "A Random Walk Through Stochastic Asset Price Models: Breakdowns and a Non-Linear Barrier Jump Model". S&P 500 (^GSPC), 2016-09-01 to 2026-09-11, N = 2,519 daily log returns. Running the scripts should reproduce these values.

| Tables | Script |
|---|---|
| 1-5 | gbm/gbm.py |
| 6 | extensions/merton.py |
| 7 | extensions/heston.py |
| 8 | extensions/variance_gamma.py |
| 9 | extensions/bates.py |
| 10 | barrier/barrier_model.py |

Run from the repository root, in this order (Bates starts from the Heston fit, the barrier model from Bates, and every table compares with GBM). Each script saves its fit to results/.

```
pip install -r requirements.txt
python -m gbm.gbm
python -m extensions.merton
python -m extensions.variance_gamma
python -m extensions.heston
python -m extensions.bates
python -m barrier.barrier_model
```

Heston, Bates and the barrier model use 100,000-particle filters and take several hours each.

**Table 1: Distributional Moments and Normality Test**

| Metric / Test | Obs (N) | Sample Value | GBM Target | Test Stat (JB) | p-value |
|---|---|---|---|---|---|
| Skewness | 2,519 | -0.6776 | 0.0000 | -- | -- |
| Kurtosis | 2,519 | 19.6907 | 3.0000 | -- | -- |
| Jarque-Bera Test | 2,519 | -- | Normality | 29,431.83 | <0.0001 |

**Table 2: Ljung-Box Test on Raw Log-Returns (Q_raw) and Squared Returns (Q_sq)**

| Lag (k) | Q_raw Stat | p-value (Raw) | Q_sq Stat | p-value (Sq) |
|---|---|---|---|---|
| 10 | 255.6934 | <0.0001 | 2,741.2237 | <0.0001 |
| 20 | 307.0394 | <0.0001 | 3,173.1028 | <0.0001 |
| 100 | 427.1381 | <0.0001 | 3,281.7427 | <0.0001 |

**Table 3: Variance Ratio Test Results**

| Series | q | VR | z_H | p_H | z_He | p_He | Type |
|---|---|---|---|---|---|---|---|
| Log-price | 2 | 0.856 | -7.233 | 4.73e-13 | -2.355 | 0.0185 | Price |
| Log-price | 5 | 0.838 | -3.721 | 0.000199 | -1.227 | 0.2198 | Price |
| Log-price | 10 | 0.811 | -2.805 | 0.00503 | -0.975 | 0.3298 | Price |
| Log-price | 20 | 0.806 | -1.959 | 0.0501 | -0.736 | 0.4615 | Price |
| Log-price | 60 | 0.624 | -2.136 | 0.0326 | -1.012 | 0.3114 | Price |
| Realized vol. (21d) | 2 | 1.209 | 10.434 | 0 | 4.008 | 6.12e-5 | Volatility |
| Realized vol. (21d) | 5 | 1.806 | 18.393 | 0 | 7.300 | 2.88e-13 | Volatility |
| Realized vol. (21d) | 10 | 2.579 | 23.375 | 0 | 9.859 | 0 | Volatility |
| Realized vol. (63d) | 2 | 1.328 | 16.265 | 0 | 4.131 | 3.61e-5 | Volatility |
| Realized vol. (63d) | 5 | 2.239 | 28.016 | 0 | 7.344 | 2.07e-13 | Volatility |
| Realized vol. (63d) | 10 | 3.509 | 36.827 | 0 | 10.371 | 0 | Volatility |
| Realized vol. (63d) | 21 | 5.458 | 43.300 | 0 | 14.114 | 0 | Volatility |

**Table 4: Augmented Dickey-Fuller Test Results**

| Series | n | Lag | ADF Stat. | p-value | 1% | 5% | Result |
|---|---|---|---|---|---|---|---|
| Log-price | 2510 | 9 | -0.480 | 0.896 | -3.433 | -2.863 | Fail to reject |
| Realized Vol. (21d) | 2471 | 27 | -5.082 | 1.52e-5 | -3.433 | -2.863 | Reject |
| Realized Vol. (63d) | 2446 | 10 | -4.468 | 0.000224 | -3.433 | -2.863 | Reject |

**Table 5: Regression and Mean Reversion Analysis Results**

| Window | n | a | b | SE_HAC | p | R^2 | kappa | theta | HL (yr) | HL (trading days) |
|---|---|---|---|---|---|---|---|---|---|---|
| 21d | 2498 | 0.001010 | 0.993123 | 0.013882 | 0 | 0.986168 | 1.738904 | 0.146882 | 0.398611 | 100.45 |
| 63d | 2456 | 0.000258 | 0.998382 | 0.006074 | 0 | 0.996798 | 0.408151 | 0.159695 | 1.698260 | 427.96 |

**Table 6: Specification diagnostics -- Merton Jump Diffusion**

| Statistic | GBM (Part 4) | Merton | Reference | p-value (Merton) |
|---|---|---|---|---|
| *Panel A: Distribution of PIT residuals* | | | | |
| Skewness | -0.678 | 0.038 | 0 | |
| Kurtosis | 19.69 | 3.730 | 3 | |
| Jarque-Bera | 29,431.8 | 56.6 | chi2(2) | <0.001 |
| *Panel B: Serial dependence (Ljung-Box)* | | | | |
| Q_raw(10) | 255.7 | 59.3 | chi2(10) | <0.001 |
| Q_raw(100) | 427.1 | 184.8 | chi2(100) | <0.001 |
| Q_sq(10) | 2,741.2 | 3,141.5 | chi2(10) | <0.001 |
| Q_sq(100) | 3,281.7 | 5,062.7 | chi2(100) | <0.001 |
| *Panel C: Variance ratios* | | | | |
| VR(2), cumulated returns / z | 0.856 | 0.929 | 1 | 0.014 |
| VR(20), cumulated returns / z | 0.806 | 0.735 | 1 | 0.055 |
| VR(10), realized vol 21d | 2.579 | 2.265 | 1 | <0.001 (MC) |
| VR(21), realized vol 63d | 5.458 | 5.613 | 1 | <0.001 (MC) |
| *Panel D: Level statistics (1,000 simulated paths)* | | | | |
| ADF, log price | -0.86 | -0.97 [-2.44, 0.81] | -0.48 (S&P) | 0.692 (MC) |
| ADF, realized vol 21d | -6.79 | -6.73 [-7.62, -5.82] | -5.08 (S&P) | <0.001 (MC) |
| OU half-life, RV 21d (days) | 13.2 | 13.6 [11.5, 16.1] | 100.5 (S&P) | <0.001 (MC) |
| OU half-life, RV 63d (days) | 40.8 | 41.0 [30.3, 56.4] | 428.0 (S&P) | <0.001 (MC) |
| *Panel E: Fit* | | | | |
| Log-likelihood | 7,692.8 | 8,123.3 | | |
| AIC | -15,381.7 | -16,236.6 | lower is better | |
| BIC | -15,370.0 | -16,207.4 | lower is better | |
| Parameters (k) | 2 | 5 | | |

Estimates (annualised, under P): alpha = 0.1412, sigma = 0.09792, lambda = 71.84, mu_J = -0.002961, sigma_J = 0.01667.

**Table 7: Specification diagnostics -- Heston Stochastic Volatility**

| Statistic | GBM (Part 4) | Heston | Reference | p-value (Heston) |
|---|---|---|---|---|
| *Panel A: Distribution of PIT residuals* | | | | |
| Skewness | -0.678 | -0.451 | 0 | |
| Kurtosis | 19.69 | 3.704 | 3 | |
| Jarque-Bera | 29,431.8 | 137.4 | chi2(2) | <0.001 |
| *Panel B: Serial dependence (Ljung-Box)* | | | | |
| Q_raw(10) | 255.7 | 18.1 | chi2(10) | 0.053 |
| Q_raw(100) | 427.1 | 112.6 | chi2(100) | 0.183 |
| Q_sq(10) | 2,741.2 | 71.5 | chi2(10) | <0.001 |
| Q_sq(100) | 3,281.7 | 207.5 | chi2(100) | <0.001 |
| *Panel C: Variance ratios* | | | | |
| VR(2), cumulated returns / z | 0.856 | 0.962 | 1 | 0.073 |
| VR(20), cumulated returns / z | 0.806 | 0.870 | 1 | 0.217 |
| VR(10), realized vol 21d | 2.579 | 1.068 | 1 | 0.207 (MC) |
| VR(21), realized vol 63d | 5.458 | 1.360 | 1 | 0.010 (MC) |
| *Panel D: Level statistics (1,000 simulated paths)* | | | | |
| ADF, log price | -0.86 | -1.03 [-2.46, 0.63] | -0.48 (S&P) | 0.604 (MC) |
| ADF, realized vol 21d | -6.79 | -5.22 [-6.18, -4.21] | -5.08 (S&P) | 0.812 (MC) |
| OU half-life, RV 21d (days) | 13.2 | 57.8 [45.5, 73.7] | 100.5 (S&P) | <0.001 (MC) |
| OU half-life, RV 63d (days) | 40.8 | 261.8 [173.6, 420.7] | 428.0 (S&P) | 0.100 (MC) |
| *Panel E: Fit* | | | | |
| Log-likelihood | 7,692.8 | 8,455.2 | | |
| AIC | -15,381.7 | -16,900.4 | lower is better | |
| BIC | -15,370.0 | -16,871.2 | lower is better | |
| Parameters (k) | 2 | 5 | | |

Estimates (annualised, under P): mu = 0.1054, kappa = 9.05, theta = 0.02821, xi = 0.63, rho = -0.669.

**Table 8: Specification diagnostics -- Variance Gamma**

| Statistic | GBM (Part 4) | VG | Reference | p-value (VG) |
|---|---|---|---|---|
| *Panel A: Distribution of PIT residuals* | | | | |
| Skewness | -0.678 | -0.127 | 0 | |
| Kurtosis | 19.69 | 3.954 | 3 | |
| Jarque-Bera | 29,431.8 | 102.3 | chi2(2) | <0.001 |
| *Panel B: Serial dependence (Ljung-Box)* | | | | |
| Q_raw(10) | 255.7 | 63.8 | chi2(10) | <0.001 |
| Q_raw(100) | 427.1 | 191.1 | chi2(100) | <0.001 |
| Q_sq(10) | 2,741.2 | 3,348.0 | chi2(10) | <0.001 |
| Q_sq(100) | 3,281.7 | 5,288.5 | chi2(100) | <0.001 |
| *Panel C: Variance ratios* | | | | |
| VR(2), cumulated returns / z | 0.856 | 0.929 | 1 | 0.016 |
| VR(20), cumulated returns / z | 0.806 | 0.774 | 1 | 0.113 |
| VR(10), realized vol 21d | 2.579 | 2.284 | 1 | <0.001 (MC) |
| VR(21), realized vol 63d | 5.458 | 5.716 | 1 | <0.001 (MC) |
| *Panel D: Level statistics (1,000 simulated paths)* | | | | |
| ADF, log price | -0.86 | -0.86 [-2.46, 0.80] | -0.48 (S&P) | 0.690 (MC) |
| ADF, realized vol 21d | -6.79 | -6.74 [-7.72, -5.93] | -5.08 (S&P) | 0.002 (MC) |
| OU half-life, RV 21d (days) | 13.2 | 13.5 [11.5, 15.9] | 100.5 (S&P) | <0.001 (MC) |
| OU half-life, RV 63d (days) | 40.8 | 40.6 [30.4, 53.8] | 428.0 (S&P) | <0.001 (MC) |
| *Panel E: Fit* | | | | |
| Log-likelihood | 7,692.8 | 8,127.5 | | |
| AIC | -15,381.7 | -16,247.1 | lower is better | |
| BIC | -15,370.0 | -16,223.8 | lower is better | |
| Parameters (k) | 2 | 4 | | |

Estimates (annualised, under P): m = 0.2541, sigma = 0.1675, theta = -0.1277, nu = 0.00469.

**Table 9: Specification diagnostics -- Bates (Stochastic Volatility with Jumps)**

| Statistic | GBM (Part 4) | Bates | Reference | p-value (Bates) |
|---|---|---|---|---|
| *Panel A: Distribution of PIT residuals* | | | | |
| Skewness | -0.678 | -0.367 | 0 | |
| Kurtosis | 19.69 | 3.521 | 3 | |
| Jarque-Bera | 29,431.8 | 84.9 | chi2(2) | <0.001 |
| *Panel B: Serial dependence (Ljung-Box)* | | | | |
| Q_raw(10) | 255.7 | 20.5 | chi2(10) | 0.025 |
| Q_raw(100) | 427.1 | 116.8 | chi2(100) | 0.121 |
| Q_sq(10) | 2,741.2 | 138.2 | chi2(10) | <0.001 |
| Q_sq(100) | 3,281.7 | 280.4 | chi2(100) | <0.001 |
| *Panel C: Variance ratios* | | | | |
| VR(2), cumulated returns / z | 0.856 | 0.959 | 1 | 0.059 |
| VR(20), cumulated returns / z | 0.806 | 0.871 | 1 | 0.231 |
| VR(10), realized vol 21d | 2.579 | 1.225 | 1 | 0.001 (MC) |
| VR(21), realized vol 63d | 5.458 | 1.649 | 1 | <0.001 (MC) |
| *Panel D: Level statistics (1,000 simulated paths)* | | | | |
| ADF, log price | -0.86 | -1.34 [-2.65, 0.31] | -0.48 (S&P) | 0.388 (MC) |
| ADF, realized vol 21d | -6.79 | -4.96 [-6.01, -4.02] | -5.08 (S&P) | 0.826 (MC) |
| OU half-life, RV 21d (days) | 13.2 | 58.1 [43.4, 75.6] | 100.5 (S&P) | <0.001 (MC) |
| OU half-life, RV 63d (days) | 40.8 | 279.1 [179.0, 449.3] | 428.0 (S&P) | 0.136 (MC) |
| *Panel E: Fit* | | | | |
| Log-likelihood | 7,692.8 | 8,463.7 | | |
| AIC | -15,381.7 | -16,911.4 | lower is better | |
| BIC | -15,370.0 | -16,864.7 | lower is better | |
| Parameters (k) | 2 | 8 | | |

Estimates (annualised, under P): mu = 0.0694, kappa = 7.472, theta = 0.02894, xi = 0.5777, rho = -0.7214, lambda = 3.255, mu_J = -0.0171, sigma_J = 0.0009119 (at its lower bound).

**Table 10: Specification diagnostics -- Supported Extension (Drawdown-Barrier Bates)**

| Statistic | GBM (Part 4) | Bates | Barrier model | Reference | p-value (Barrier model) |
|---|---|---|---|---|---|
| *Panel A: Distribution of PIT residuals* | | | | | |
| Skewness | -0.678 | -0.367 | -0.283 | 0 | |
| Kurtosis | 19.69 | 3.521 | 3.373 | 3 | |
| Jarque-Bera | 29,431.8 | 84.9 | 48.2 | chi2(2) | <0.001 |
| *Panel B: Serial dependence (Ljung-Box)* | | | | | |
| Q_raw(10) | 255.7 | 20.5 | 21.6 | chi2(10) | 0.017 |
| Q_raw(100) | 427.1 | 116.8 | 117.3 | chi2(100) | 0.114 |
| Q_sq(10) | 2,741.2 | 138.2 | 150.8 | chi2(10) | <0.001 |
| Q_sq(100) | 3,281.7 | 280.4 | 287.7 | chi2(100) | <0.001 |
| *Panel C: Variance ratios* | | | | | |
| VR(2), cumulated returns / z | 0.856 | 0.959 | 0.955 | 1 | 0.042 |
| VR(20), cumulated returns / z | 0.806 | 0.871 | 0.852 | 1 | 0.169 |
| VR(10), realized vol 21d | 2.579 | 1.225 | 1.220 | 1 | 0.001 (MC) |
| VR(21), realized vol 63d | 5.458 | 1.649 | 1.715 | 1 | <0.001 (MC) |
| *Panel D: Level statistics (1,000 simulated paths)* | | | | | |
| ADF, log price | -0.86 | -1.34 | -1.31 [-2.62, 0.21] | -0.48 (S&P) | 0.346 (MC) |
| ADF, realized vol 21d | -6.79 | -4.96 | -4.90 [-5.87, -3.89] | -5.08 (S&P) | 0.756 (MC) |
| OU half-life, RV 21d (days) | 13.2 | 58.1 | 45.5 [33.3, 61.9] | 100.5 (S&P) | <0.001 (MC) |
| OU half-life, RV 63d (days) | 40.8 | 279.1 | 217.1 [141.9, 352.2] | 428.0 (S&P) | 0.046 (MC) |
| *Panel E: Fit* | | | | | |
| Log-likelihood | 7,692.8 | 8,463.7 | 8,468.1 | | |
| AIC | -15,381.7 | -16,911.4 | -16,914.3 | lower is better | |
| BIC | -15,370.0 | -16,864.7 | -16,850.1 | lower is better | |
| Parameters (k) | 2 | 8 | 11 | | |

Estimates (annualised, under P): mu = 0.06349, kappa = 7.859, theta = 0.02747, xi = 0.5714, rho = -0.7298, lambda_0 = 3.165, mu_J,0 = 0.007504, sigma_J = 0.0009119, a = 1.335, b = -0.05653, gamma = 18.7. Barrier B = 0.8 x trailing 252-day high; g_t >= 0.39 on every sample day, so in sample lambda_t spans 3.7-4.5 jumps per year and mu_J,t spans -1.5% to -4.9%. 73 trading days fall at or below the barrier.

Notes for Tables 6-10: "GBM (Part 4)" repeats the Section 4 statistic (raw returns are GBM's PIT residuals); the model column is the same statistic on that model's PIT residuals z_t = Phi^-1(F_t(r_t)), i.i.d. N(0,1) under correct specification. p-values are asymptotic except (MC): Panel C realized-vol rows against 2,000 i.i.d. N(0,1) series; Panel D the S&P value against 1,000 paths simulated from the fitted model. Variance-ratio p-values use the heteroskedasticity-robust z*. Panel D: GBM and comparison columns are simulated medians, model 90% band in brackets; OU p-values are computed on the AR(1) coefficient b. Heston, Bates and barrier-model log-likelihoods are particle-filter estimates (100,000 particles, mean of 3 seeds).
