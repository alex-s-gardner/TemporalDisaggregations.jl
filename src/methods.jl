

"""
    DisaggregationMethod

Abstract supertype for temporal disaggregation algorithms.
Subtypes hold algorithm-specific parameters with sensible defaults.
"""
abstract type DisaggregationMethod end

"""
    Spline(; smoothness=1e-3, penalty_order=3, tension=0.0)

Quartic B-spline antiderivative fit with P-spline regularization.

Models the antiderivative F(t) as a B-spline so that F(t2ᵢ) − F(t1ᵢ) equals the
observed area for each interval; the instantaneous signal is x(t) = F′(t).

Knots are placed at fixed monthly spacing (~12 per year) to maintain consistent
smoothness behavior. The spline is evaluated at the user-requested `output_period`,
which can be finer (e.g., daily) or coarser (e.g., quarterly) than the knot spacing.

The `:std` layer in the returned `DimStack` is a spatially-varying sandwich standard
deviation: lower where observations are dense, higher where they are sparse.

# Keywords
- `smoothness::Float64 = 1e-1`: Regularization strength λ (larger = smoother).
  **Note**: When using robust losses like `L1DistLoss()`, the same `smoothness` value
  may produce smoother results than with `L2DistLoss()`. This is expected behavior —
  tune `smoothness` separately for each loss function.
- `penalty_order::Int = 3`: Order of the difference penalty.
- `tension::Float64 = 0.0`: Tension penalty strength (0 = standard P-spline).
"""
@kwdef struct Spline <: DisaggregationMethod
    smoothness::Float64              = 1e-1
    penalty_order::Int               = 3
    tension::Float64                 = 0.0
end

"""
    Sinusoid(; smoothness_interannual=1e-2)

Parametric model: mean + trend + per-year anomalies + annual sinusoid.
All interval integrals are solved analytically (no quadrature) — the fastest method.

The `:std` layer in the returned `DimStack` is a spatially-varying sandwich standard
deviation: lower where observations are dense, higher where they are sparse.

# Keywords
- `smoothness_interannual::Float64 = 1e-2`: Ridge penalty on inter-annual anomalies γ.
"""
@kwdef struct Sinusoid <: DisaggregationMethod
    smoothness_interannual::Float64  = 1e-2
end

"""
    PiecewiseLinear(; smoothness=1e-6, n_knots=0, tension=0.0)

Piecewise linear disaggregation using linear hat functions with first-order
difference penalties. Preserves sharp corners and triangular patterns.

Ideal for time series with monotonic increase/decrease patterns (triangular peaks)
superimposed on trends. Unlike `Spline` (smooth curves) or `GP` (infinitely smooth),
this method produces C⁰ continuous signals with piecewise constant slopes, preserving
sharp transitions.

The signal x(t) = Σₖ θₖ φₖ(t) where φₖ(t) are linear hat functions (tent functions)
on a uniform knot grid. The first-order difference penalty ‖D₁ θ‖² penalizes slope
changes between adjacent segments, encouraging piecewise linear patterns.

All interval integrals are solved analytically (no quadrature) — comparable speed to
`Spline` but preserves triangular peaks that `Spline` and `GP` would over-smooth.

The `:std` layer in the returned `DimStack` is a spatially-varying sandwich standard
deviation: lower where observations are dense, higher where they are sparse.

# Keywords
- `smoothness::Float64 = 1e-6`: Regularization strength λ. Lower values preserve
  sharper corners. Scaled by ‖C'C‖/n to be dimensionless across datasets. Default
  is much lower than `Spline` to minimize smoothing of triangular patterns.
- `n_knots::Int = 0`: Number of knots for basis functions. If 0 (default),
  auto-computed from time span and `output_period` (approximately one knot per period).
- `tension::Float64 = 0.0`: Tension parameter ∈ [0,1]. Blends first-order
  penalty (0.0) toward pure interpolation (1.0). Higher values reduce smoothing.

# Examples
```julia
# Preserve sharp triangular patterns with minimal smoothing
result = disaggregate(PiecewiseLinear(smoothness=1e-8), values, t1, t2)

# Auto knot spacing tied to monthly output
result = disaggregate(PiecewiseLinear(), values, t1, t2; output_period=Month(1))

# Weekly output with explicit knot count
result = disaggregate(PiecewiseLinear(n_knots=200), values, t1, t2; output_period=Week(1))
```
"""
@kwdef struct PiecewiseLinear <: DisaggregationMethod
    smoothness::Float64 = 1e-6
    n_knots::Int = 0
    tension::Float64 = 0.0
end

"""
    GP(; kernel=with_lengthscale(Matern52Kernel(), 1/6), obs_noise=1.0, n_quad=5)

Sparse inducing-point Gaussian Process (DTC approximation) on a monthly grid.
Supports arbitrary `KernelFunctions.jl` kernels (sums, products, periodic, etc.).

The `:std` layer in the returned `DimStack` is a spatially-varying sandwich standard
deviation: lower where observations are dense, higher where they are sparse.

# Keywords
- `kernel`: Any `KernelFunctions.jl` kernel. Default: Matérn-5/2 with 2-month lengthscale.
- `obs_noise::Float64 = 1.0`: Observation noise variance σ² **in the same units as y²**.
  Controls the GP posterior smoothness: smaller values → tighter fit to observations.
- `n_quad::Int = 5`: Gauss-Legendre quadrature points per interval.
"""
@kwdef struct GP{K} <: DisaggregationMethod
    kernel::K                        = with_lengthscale(Matern52Kernel(), 1/6)
    obs_noise::Float64               = 1.0
    n_quad::Int                      = 5
end

function Base.show(io::IO, m::GP)
    print(io, "GP(obs_noise=$(m.obs_noise), n_quad=$(m.n_quad), kernel=…)")
end

function Base.show(io::IO, ::MIME"text/plain", m::GP)
    println(io, "GP")
    println(io, "  kernel:    ", m.kernel)
    println(io, "  obs_noise: ", m.obs_noise)
    print(io,   "  n_quad:    ", m.n_quad)
end


