# Model, filtering equations and checks

## Continuous-noise convention

State `x=[px,py,vx,vy]` uses metres and metres/second. The transition is `F=[[I,dt I],[0,I]]`. Independent continuous white accelerations on the two axes have spectral density q in m²/s³. Integration over a time interval yields

```text
Q = q * [[dt^3/3 I, dt^2/2 I],
         [dt^2/2 I,     dt I]]
```

This satisfies `Q(t+s)=F(s)Q(t)F(s)^T+Q(s)`, which is tested independently. It differs from a single constant random acceleration over each interval, whose position variance has a `dt^4/4` factor. The synthetic truth samples increments directly from this integrated covariance.

The study uses dt=0.5 s, true q=0.18 m²/s³, prior mean `[0,0,1,0.35]` and prior covariance `diag(4,4,1,1)`. Initial truth is sampled from that same prior. Measurement matrix `H=[I,0]` and covariance `R=[[2.25,0.45],[0.45,1.44]]` in m² are fixed. Process increments, initial-state error and measurement errors are independent.

## Recursion

```text
m- = F m
P- = F P F^T + Q
y  = z - H m-
S  = H P- H^T + R
solve S K^T = H P-
m+ = m- + K y
P+ = (I-KH) P- (I-KH)^T + K R K^T
```

All linear algebra uses NumPy solves, multiplication and symmetric eigendecomposition for validation. There is no explicit matrix inverse. The Joseph covariance expression avoids the direct subtraction used in the simpler covariance formula; it does not make arbitrary floating-point input immune to numerical problems.

If an observation is missing, the posterior is the prediction. For a present observation, NIS = `y^T solve(S,y)`. The optional gate accepts when NIS is **at most** the threshold, including equality. A rejected observation leaves the prediction unchanged and retains its innovation record.

The gate ignores information contained in the rejection event itself. The resulting mean/covariance are therefore a practical gated approximation, not the exact Gaussian posterior conditioned on a selection event.

## Validation boundaries

The generic state dimension and observation dimension are each 1–8. Arrays must have correct shapes and finite entries. Covariance symmetry and nonnegative eigenvalues are validated with a scale-dependent tolerance of `1e-10 * max(1,max(abs(P)))`. Matrices are symmetrized after acceptance and after updates. This finite tolerance permits tiny negative roundoff eigenvalues; there is no assertion of exact PSD for all floating-point inputs. R and S require a strictly positive minimum computed eigenvalue.

The tracking wrapper requires four states, `0 < dt <= 10` and `0 <= q <= 100`. Gate thresholds must be finite and positive. The study is well-conditioned within these ranges, but dimensions/bounds do not guarantee stable computation under every adversarial magnitude or condition number. Arrays are copied and marked read-only at estimate construction; this prevents accidental aliasing, not hostile Python mutation.

## Independent checks

The scalar test uses prior mean 2, variance 4, observation 5 and measurement variance 9. Posterior mean is 38/13, variance 36/13 and NIS 9/13.

For each of 24 small randomized problems, a separate implementation stacks prior, process and observation equations into a whitened least-squares system. It solves all states jointly, using Cholesky whitening, `numpy.linalg.lstsq` for the mean and QR triangular solves for the final covariance block. It never calls the filter. Its last state must agree with the sequential filter to tight tolerance, including omitted measurements.

The archive verifier separately reconstructs transitions and covariances and checks updates with the conditional-Gaussian covariance `P- − K S K^T`. This different algebraic expression cross-checks the saved Joseph results.

## Primary references

- Simo Särkkä, [Bayesian Filtering and Smoothing](https://users.aalto.fi/~ssarkka/pub/cup_book_online_20131111.pdf), Cambridge University Press, 2013, Chapters 3–4: batch/recursive Gaussian estimation and Kalman filtering.
- Renato Zanetti and Kyle J. DeMars, [Joseph Formulation of Unscented and Quadrature Filters with Application to Consider States](https://sites.utexas.edu/near/wp-content/uploads/sites/6030/2017/04/CUKF_ver06.pdf), introduction: Joseph covariance expression and numerical motivation. This project implements the linear case only.
- [FilterPy author documentation: continuous white-noise discretization](https://filterpy.readthedocs.io/en/latest/_modules/filterpy/common/discretization.html): confirms the integrated continuous-noise convention and its distinction from discrete constant acceleration. FilterPy is a reference only; it is not installed or used by this implementation.
