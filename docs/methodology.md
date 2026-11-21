# Synthetic experiment and statistical interpretation

## Predeclared cases

The illustrative seed is 20261121. It is a single visualization example, not evidence that every trajectory behaves similarly. The Monte Carlo seed is 661024; NumPy SeedSequence spawns distinct run seeds. Each run samples a new initial state, process noise and observation noise. All five configurations within a run share the same latent trajectory and clean observations:

| Configuration | Filter q | Observations | Gate |
|---|---:|---|---|
| matched | 0.18 | every step | none |
| low_process_noise | 0.006 | every step | none |
| sparse_measurements | 0.18 | every third step; steps 25–36 missing | none |
| outliers_ungated | 0.18 | every step; fixed faults | none |
| outliers_gated | 0.18 | same fixed faults | 9.21034 |

Faults add `[11,-9]` metres at steps 18 and 48 and `[-11,9]` at step 36. They are deliberately injected errors, not a claim about any physical sensor's outlier distribution. The gate is the nominal χ²₂ 99% threshold `-2 log(0.01)`.

## Independent unit of analysis

The main comparison uses step 60, at 30 seconds, once per independent run. The same-run pairing across configurations is retained, but configurations are not counted as extra independent samples. Time-series plots are explicitly descriptive. No coverage test treats all 60 times as independent trials.

Position RMSE is `sqrt(mean(||position error||²))`; velocity RMSE is defined analogously. Full-state NEES is `error^T solve(P,error)`. A correctly specified ungated four-state Gaussian model has nominal expected NEES 4. Position containment uses `error_pos^T solve(P_pos,error_pos) <= -2 log(0.05)`, the 95% χ²₂ contour.

Coverage intervals are Wilson score 95% intervals over independent final-time Bernoulli indicators. They quantify uncertainty in this Monte Carlo coverage rate, not an interval for a specific robot position. NEES standard errors use the empirical standard deviation across independent runs divided by sqrt(run count). Chart NEES bars are **one standard error**, not 95% intervals.

NIS is stored before acceptance/rejection for every observed step. Only the fixed-time statistic is summarized across runs. Accepted-only NIS is selected by the gate and should not be compared naively with an untruncated χ² reference. Prior gating decisions also alter the current conditional distribution; gated consistency numbers are descriptive, even when all final-time candidates are included.

The matched/sparse cases have correctly specified Gaussian dynamics. The underestimated-Q and contaminated/gated cases violate some of those nominal conditions. Their reference lines are diagnostic context, not automatic confidence tests.

The final-time results are retained even though gating has slightly worse final RMSE than the ungated contaminated case in this run. The last fault occurs at step 48, allowing ungated recovery before the final evaluation. There is no tuning on the reported endpoint to force a preferred result.

## Local timing

Fifteen calls per configuration follow two untimed warmups. Configuration order is shuffled with Python seed 14992. Each observation measures all 60 filtering steps on the same illustrative input, including finite/covariance checks and construction of the Python record dictionaries/lists. Simulation, JSON encoding, file writes, plotting and archive verification are excluded.

These are local wall-clock observations using perf_counter_ns, without CPU pinning or process isolation. They are not comparable robot latency limits or an optimized-filter benchmark. NumPy, Matplotlib, Python and host details are retained.

## Limits

Synthetic linear Gaussian position tracking does not cover orientation, nonlinear dynamics, data association, asynchronous clocks, systematic sensor bias or online parameter estimation. Marginal position ellipses at selected times do not form a simultaneous whole-path confidence region. Rejected observations may be legitimate changes in motion; a gate is not an outlier-cause detector.

No project output represents field testing or manual development footage. Implementation, analysis and documentation were AI-assisted.
