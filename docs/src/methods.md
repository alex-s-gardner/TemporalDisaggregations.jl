# Methods

All four methods share the same interface and return type. Switch methods by passing a different algorithm struct as the first argument to `disaggregate`.

## Comparison

Benchmarks: 20-year span, `output_period=Week(1)`, 8 threads (Julia 1.12). Time = minimum over 2–3 runs. Memory = dominant working-set matrix (n × matrix\_width × 8 B).

| Method | Pros | Cons | n = 10k | n = 100k | n = 1M |
|--------|------|------|:-------:|:--------:|:------:|
| `Spline` | No kernel required; optional tension suppresses oscillation near sparse gaps | Design matrix O(n × n\_knots); can oscillate without tension | 12 ms / 19 MB | 103 ms / 192 MB | 807 ms / 1.9 GB |
| `Sinusoid` | Analytical integrals (no quadrature); interpretable parameters (amplitude, phase, trend, anomalies); lowest peak memory | Assumes annual periodicity; poor fit for non-sinusoidal signals | 61 ms / 2 MB | 133 ms / 19 MB | 2.4 s / 192 MB |
| `GP` | Arbitrary KernelFunctions.jl kernels; most flexible | O(n·m·q + m³) Cholesky — memory-limited above n ≈ 50 000 at weekly output | 2.0 s / 195 MB | 13.3 s / 1.9 GB | — (>8 GB) |
| `GPKF` | O(n·d²) Kalman filter; exact posterior (no inducing approximation); scales to n=1M | TemporalGPs-compatible kernels only; no `PeriodicKernel` | 24 ms / 1.2 MB | 241 ms / 12 MB | 2.3 s / 120 MB |

## B-spline (`Spline`)

Fits a smooth curve whose running averages match the observations. A regularisation parameter controls smoothness; an optional tension penalty suppresses oscillation near sparse regions.

```julia
result = disaggregate(Spline(
    smoothness    = 1e-3,       # larger = smoother
    tension       = 0.0,        # > 0 suppresses oscillation
    penalty_order = 3,          # order of difference penalty
), y, t1, t2; loss_norm = :L2)
```

**Uncertainty:** Spatially-varying sandwich std — lower where observations are dense, higher where they are sparse.

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

**Uncertainty:** Spatially-varying sandwich std — lower where observations are dense, higher where they are sparse.

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

**Uncertainty:** Spatially-varying sandwich std — lower where observations are dense, higher where they are sparse.

![GP posterior mean and 2σ band](./assets/gp_detail.png)

## Gaussian Process via Kalman Filter (`GPKF`)

Uses a state-space GP solved with a Kalman filter (TemporalGPs.jl) — O(n·d²) complexity, no large Cholesky, scales to n=1e6. Each interval observation is expanded into `n_quad` Gauss-Legendre pseudo-points with matching precision. Requires a TemporalGPs-compatible kernel; do not use `PeriodicKernel` (use `ApproxPeriodicKernel{N}` instead).

```julia
using KernelFunctions

k = 12.0^2 * with_lengthscale(Matern52Kernel(), 1.0) +
     4.0^2 * with_lengthscale(Matern52Kernel(), 2.0)

result = disaggregate(GPKF(
    kernel    = k,
    obs_noise = 4.0,    # observation noise variance σ²
    n_quad    = 5,      # Gauss-Legendre quadrature points per interval
), y, t1, t2)
```

**Uncertainty:** Spatially-varying sandwich std — lower where observations are dense, higher where they are sparse.

![GPKF posterior mean and 2σ band](./assets/gpkf_detail.png)

## Uncertainty

All four methods return the same type of `std` — a spatially-varying sandwich standard deviation:

```
std(t*) = σ̂ · sqrt(q(t*))
```

where `σ̂` is the weighted residual RMS of predicted vs. observed interval averages
(`sqrt(Σ wᵢ rᵢ² / Σ wᵢ)`, with `rᵢ = yᵢ − ŷᵢ`) and `q(t*)` is a dimensionless
coverage factor derived from the method's hat vector at time `t*`:

- **Dense observation coverage** → `q(t*) < 1` → `std(t*) < σ̂`
- **Sparse observation coverage** → `q(t*) > 1` → `std(t*) > σ̂`

This makes `std` comparable across methods and automatically reflects the temporal
distribution of the input observations. When using `loss_norm = :L1`, both `σ̂` and
`q(t*)` are computed from the final IRLS solution.
