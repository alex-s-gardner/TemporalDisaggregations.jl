

"""
    DisaggregationMethod

Abstract supertype for temporal disaggregation algorithms.
Subtypes hold algorithm-specific parameters with sensible defaults.
"""
abstract type DisaggregationMethod end

"""
    Spline(; smoothness=1e-3, n_knots=nothing, penalty_order=3, tension=0.0)

Quartic B-spline antiderivative fit with P-spline regularization.

Models the antiderivative F(t) as a B-spline so that F(t2ᵢ) − F(t1ᵢ) equals the
observed area for each interval; the instantaneous signal is x(t) = F′(t).

The `:std` layer in the returned `DimStack` is constant across the output grid and equals
the residual standard deviation of predicted vs. observed interval averages: `std(y .- ŷ)`.

# Keywords
- `smoothness::Float64 = 1e-3`: Regularization strength λ (larger = smoother).
- `n_knots::Union{Int,Nothing} = nothing`: Number of knots (`nothing` = auto monthly, `0` = dense).
- `penalty_order::Int = 3`: Order of the difference penalty.
- `tension::Float64 = 0.0`: Tension penalty strength (0 = standard P-spline).
"""
@kwdef struct Spline <: DisaggregationMethod
    smoothness::Float64              = 1e-3
    n_knots::Union{Int,Nothing}      = nothing
    penalty_order::Int               = 3
    tension::Float64                 = 0.0
end

"""
    Sinusoid(; smoothness_interannual=1e-2)

Parametric model: mean + trend + per-year anomalies + annual sinusoid.
All interval integrals are solved analytically (no quadrature) — the fastest method.

The `:std` layer in the returned `DimStack` is constant across the output grid and equals
the residual standard deviation of predicted vs. observed interval averages: `std(y .- ŷ)`.

# Keywords
- `smoothness_interannual::Float64 = 1e-2`: Ridge penalty on inter-annual anomalies γ.
"""
@kwdef struct Sinusoid <: DisaggregationMethod
    smoothness_interannual::Float64  = 1e-2
end

"""
    GP(; kernel=with_lengthscale(Matern52Kernel(), 1/6), obs_noise=1.0, n_quad=5)

Sparse inducing-point Gaussian Process (DTC approximation) on a monthly grid.
Supports arbitrary `KernelFunctions.jl` kernels (sums, products, periodic, etc.).

# Keywords
- `kernel`: Any `KernelFunctions.jl` kernel. Default: Matérn-5/2 with 2-month lengthscale.
  The kernel's **output scale must match the variance of your data**. A unit-variance kernel
  (e.g. `SqExponentialKernel()`) implicitly assumes signal amplitude ≈ 1; for data with
  standard deviation σ, use `σ^2 * SqExponentialKernel()` (or the equivalent for any kernel).
- `obs_noise::Float64 = 1.0`: Observation noise variance σ² **in the same units as y²**.
  Set to `var(y) * snr⁻¹` where `snr` is the expected signal-to-noise ratio, or tune
  relative to the kernel's output scale.
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
