# TemporalDisaggregations.jl

[![Build Status](https://github.com/alex-s-gardner/TemporalDisaggregations.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/alex-s-gardner/TemporalDisaggregations.jl/actions/workflows/CI.yml?query=branch%3Amain)

Reconstruct instantaneous time series from **interval-averaged observations** — measurements that represent the average of a signal over a time window rather than a point-in-time snapshot.

![Overview of all three methods](docs/images/overview.png)

## The Problem

Many real-world measurements are temporal averages rather than point observations. Examples include:
- **Remote sensing**: satellite image-pair velocities averaged over a revisit period
- **Hydrology**: stream-gauge discharge totals over reporting intervals
- **Climatology**: monthly or seasonal summaries of daily observations
- **Finance**: period-average prices or returns

When intervals are irregular, overlapping, or sparse, standard interpolation fails. `TemporalDisaggregations.jl` solves the inverse problem: given *n* interval averages `yᵢ ≈ (1/Δtᵢ) ∫_{t1ᵢ}^{t2ᵢ} x(t) dt`, recover the instantaneous signal `x(t)` on a regular output grid with uncertainty estimates.

## Installation

```julia
using Pkg
Pkg.add("TemporalDisaggregations")
```

## Quick Start

```julia
using TemporalDisaggregations
using Dates

# Your data: interval-averaged observations
y  = [2.3, 1.8, 3.1, 2.7, ...]          # observed averages
t1 = [Date(2020,1,5), Date(2020,2,3), ...]  # interval start dates
t2 = [Date(2020,1,28), Date(2020,3,10), ...]  # interval end dates

# Reconstruct on a monthly grid (default)
result = disaggregate(y, t1, t2)

# Access results
dates  = collect(dims(result.values, Ti))   # Vector{Date}
values = Array(result.values)               # Vector{Float64} — posterior mean
stds   = Array(result.std)                  # Vector{Float64} — posterior std
```

## Methods

All three methods share the same interface and return type. Switch methods with a single keyword argument.

### B-spline (`method = :spline`)

Models the antiderivative `F(t)` of the instantaneous signal as a quartic B-spline, constrained so that `F(t2ᵢ) − F(t1ᵢ)` matches each observed area. The instantaneous signal is `x(t) = F′(t)`. P-spline regularisation controls smoothness; an optional tension penalty suppresses oscillation near sparse regions.

```julia
result = disaggregate(y, t1, t2;
    method        = :spline,
    smoothness    = 1e-3,       # larger = smoother
    tension       = 0.0,        # > 0 suppresses oscillation
    penalty_order = 3,          # order of difference penalty
    loss_norm     = :L2,        # or :L1 for robustness to blunders
)
```

**Uncertainty:** Frequentist P-spline confidence band derived from the hat-matrix trace of the regularised normal equations.

### Sinusoid (`method = :sinusoid`)

Fits the parametric model

```
x(t) = μ + β·(t − t̄) + γ(year) + A·sin(2πt) + B·cos(2πt)
```

where `μ` is the mean, `β` is a linear trend, `γ(year)` is a per-year anomaly, and `A`, `B` are annual seasonal amplitudes. The design matrix is constructed analytically (exact interval integrals, no quadrature) — the fastest method. Fitted parameters are accessible in the result metadata.

```julia
result = disaggregate(y, t1, t2;
    method                 = :sinusoid,
    smoothness_interannual = 1e-2,   # ridge penalty on year-to-year anomalies
    loss_norm              = :L2,
)

# Fitted parameters
using DimensionalData: metadata
md = metadata(result)
md[:amplitude]    # seasonal amplitude √(A²+B²)
md[:phase]        # peak time within year (fraction)
md[:trend]        # linear trend (units/year)
md[:interannual]  # Dict{Int,Float64} of per-year anomalies
```

**Uncertainty:** WLS covariance propagation from the regularised normal equations.

### Gaussian Process (`method = :gp`)

Models the signal as a GP prior specified by a `KernelFunctions.jl` kernel. Each observation is the integral of the GP over its interval, approximated by Gauss-Legendre quadrature. A sparse inducing-point (DTC) approximation on a monthly grid reduces computation from O(n³) to O(m³) where m ≈ 12·years.

```julia
using KernelFunctions

k = 15.0^2 * PeriodicKernel(r=[0.5]) * with_lengthscale(Matern52Kernel(), 3.0) +
     5.0^2 * with_lengthscale(Matern52Kernel(), 2.0)

result = disaggregate(y, t1, t2;
    method    = :gp,
    kernel    = k,
    obs_noise = 4.0,    # observation noise variance σ²
    n_quad    = 5,      # Gauss-Legendre quadrature points per interval
    loss_norm = :L2,
)
```

**Uncertainty:** Exact GP posterior standard deviation (Bayesian credible interval).

![GP posterior mean and 2σ band](docs/images/gp_detail.png)

## Common Options

### Output Resolution

```julia
# Daily output
result = disaggregate(y, t1, t2; output_step = Day(1))

# Weekly output
result = disaggregate(y, t1, t2; output_step = Week(1))
```

### Robust L1 Loss

All methods support `loss_norm = :L1` for robustness to blunders (outliers). L1 is solved via Iteratively Reweighted Least Squares (IRLS):

```julia
result = disaggregate(y, t1, t2; method = :gp, loss_norm = :L1, obs_noise = 4.0)
```

## Return Type

All methods return a `DimStack` (from [DimensionalData.jl](https://github.com/rafaqz/DimensionalData.jl)) with two layers:

```julia
result.values    # DimArray — posterior mean on the output Ti(dates) grid
result.std       # DimArray — posterior standard deviation (same grid)
```

Standard access patterns:

```julia
using DimensionalData: dims, Ti, metadata

dates  = collect(dims(result.values, Ti))   # Vector{Date}
values = Array(result.values)               # plain Vector{Float64}
stds   = Array(result.std)                  # plain Vector{Float64}
meta   = metadata(result)                   # Dict with method-specific params
```

DimensionalData integration enables direct use with plotting libraries, `DimArray` arithmetic, and labelled array operations.

## Dependencies

- [BasicBSpline.jl](https://github.com/hyrodium/BasicBSpline.jl) — B-spline basis evaluation (spline method)
- [AbstractGPs.jl](https://github.com/JuliaGaussianProcesses/AbstractGPs.jl) + [KernelFunctions.jl](https://github.com/JuliaGaussianProcesses/KernelFunctions.jl) — GP prior and kernel definitions (GP method)
- [FastGaussQuadrature.jl](https://github.com/JuliaApproximation/FastGaussQuadrature.jl) — Gauss-Legendre quadrature (GP method)
- [DimensionalData.jl](https://github.com/rafaqz/DimensionalData.jl) — labelled array return type (all methods)
