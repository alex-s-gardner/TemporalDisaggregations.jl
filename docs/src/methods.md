# Methods

All three methods share the same interface and return type. Switch methods by passing a different algorithm struct as the first argument to `disaggregate`.

## B-spline (`Spline`)

Fits a smooth curve whose running averages match the observations. A regularisation parameter controls smoothness; an optional tension penalty suppresses oscillation near sparse regions.

```julia
result = disaggregate(Spline(
    smoothness    = 1e-3,       # larger = smoother
    tension       = 0.0,        # > 0 suppresses oscillation
    penalty_order = 3,          # order of difference penalty
), y, t1, t2; loss_norm = :L2)
```

**Uncertainty:** Confidence band derived from how strongly regularisation constrains the fit.

![B-spline reconstruction](./assets/spline_detail.png)

## Tension-spline (`Spline` with `tension > 0`)

Adding tension stiffens the curve in data-sparse regions — think of pulling the spline taut like a guitar string. It suppresses oscillation while preserving fidelity where observations are dense.

```julia
result = disaggregate(Spline(
    smoothness = 1e-3,
    tension    = 25.0,    # 0.5–1 moderate; 5–10 near piecewise-linear; >20 strongly stiffened
), y, t1, t2)
```

![Tension-spline reconstruction](./assets/tension_spline_detail.png)

## Sinusoid (`Sinusoid`)

Fits the parametric model:

```
x(t) = μ + β·(t − t̄) + γ(year) + A·sin(2πt) + B·cos(2πt)
```

where `μ` is the mean, `β` is a linear trend, `γ(year)` is a per-year anomaly, and `A`, `B` are annual seasonal amplitudes. All integrals are solved analytically — making this the fastest method.

```julia
result = disaggregate(Sinusoid(
    smoothness_interannual = 1e-2,   # ridge penalty on year-to-year anomalies
), y, t1, t2)

# Fitted parameters
using DimensionalData: metadata
md = metadata(result)
md[:amplitude]    # seasonal amplitude √(A²+B²)
md[:phase]        # peak time within year (fraction)
md[:trend]        # linear trend (units/year)
md[:interannual]  # Dict{Int,Float64} of per-year anomalies
```

**Uncertainty:** Propagated from fitted model coefficients via weighted least squares.

![Sinusoid reconstruction](./assets/sinusoid_detail.png)

## Gaussian Process (`GP`)

Models the signal as a Gaussian Process — a flexible probabilistic model encoding correlations through time. A sparse approximation keeps computation fast even for long records. Specify the correlation structure via a [KernelFunctions.jl](https://github.com/JuliaGaussianProcesses/KernelFunctions.jl) kernel.

```julia
using KernelFunctions

k = 15.0^2 * PeriodicKernel(r=[0.5]) * with_lengthscale(Matern52Kernel(), 3.0) +
     5.0^2 * with_lengthscale(Matern52Kernel(), 2.0)

result = disaggregate(GP(
    kernel    = k,
    obs_noise = 4.0,    # observation noise variance σ²
    n_quad    = 5,      # Gauss-Legendre quadrature points per interval
), y, t1, t2)
```

**Uncertainty:** Full GP posterior standard deviation — a true probabilistic credible interval given the chosen kernel.

![GP posterior mean and 2σ band](./assets/gp_detail.png)

## Uncertainty Comparison

!!! warning
    `std` values are not directly comparable across methods. Each method derives uncertainty differently:

| Method | What `std` measures | Key caveat |
|--------|---------------------|------------|
| **GP** | True Bayesian uncertainty from the GP model | Depends on your choice of kernel and `obs_noise` |
| **Spline** | How strongly regularisation constrains the fit | Controlled by `smoothness`; does not account for uncertainty in the smoothness level itself |
| **Sinusoid** | Uncertainty in the fitted seasonal parameters | Only valid if the true signal is well-described by mean + trend + sinusoid |

When using `loss_norm = :L1`, `std` is approximate — computed from the final reweighted system, not from L1 theory.
