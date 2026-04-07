

"""
    DisaggregationMethod

Abstract supertype for temporal disaggregation algorithms.
Subtypes hold algorithm-specific parameters with sensible defaults.
"""
abstract type DisaggregationMethod end

"""
    Spline(; smoothness=1e-3, n_knots=nothing, penalty_order=3, tension=0.0, huber_delta=1.345)

Quartic B-spline antiderivative fit with P-spline regularization.

Models the antiderivative F(t) as a B-spline so that F(t2ᵢ) − F(t1ᵢ) equals the
observed area for each interval; the instantaneous signal is x(t) = F′(t).

The `:std` layer in the returned `DimStack` is a spatially-varying sandwich standard
deviation: lower where observations are dense, higher where they are sparse.

# Keywords
- `smoothness::Float64 = 1e-1`: Regularization strength λ (larger = smoother).
- `n_knots::Union{Int,Nothing} = nothing`: Number of knots (`nothing` = auto monthly, `0` = dense).
- `penalty_order::Int = 3`: Order of the difference penalty.
- `tension::Float64 = 0.0`: Tension penalty strength (0 = standard P-spline).
- `huber_delta::Float64 = 1.345`: Threshold parameter for Huber loss (only used when
  `loss_norm = :Huber`). Controls transition from quadratic (L2) to linear (L1) behavior.
  Huber's recommendation: δ = 1.345 for 95% efficiency at the normal distribution.
"""
@kwdef struct Spline <: DisaggregationMethod
    smoothness::Float64              = 1e-1
    n_knots::Union{Int,Nothing}      = nothing
    penalty_order::Int               = 3
    tension::Float64                 = 0.0
    huber_delta::Float64             = 1.345

    function Spline(smoothness, n_knots, penalty_order, tension, huber_delta)
        huber_delta > 0 || throw(ArgumentError("huber_delta must be positive; got $huber_delta."))
        new(smoothness, n_knots, penalty_order, tension, huber_delta)
    end
end

"""
    Sinusoid(; smoothness_interannual=1e-2, huber_delta=1.345)

Parametric model: mean + trend + per-year anomalies + annual sinusoid.
All interval integrals are solved analytically (no quadrature) — the fastest method.

The `:std` layer in the returned `DimStack` is a spatially-varying sandwich standard
deviation: lower where observations are dense, higher where they are sparse.

# Keywords
- `smoothness_interannual::Float64 = 1e-2`: Ridge penalty on inter-annual anomalies γ.
- `huber_delta::Float64 = 1.345`: Threshold parameter for Huber loss (only used when
  `loss_norm = :Huber`). Controls transition from quadratic (L2) to linear (L1) behavior.
  Huber's recommendation: δ = 1.345 for 95% efficiency at the normal distribution.
"""
@kwdef struct Sinusoid <: DisaggregationMethod
    smoothness_interannual::Float64  = 1e-2
    huber_delta::Float64             = 1.345

    function Sinusoid(smoothness_interannual, huber_delta)
        huber_delta > 0 || throw(ArgumentError("huber_delta must be positive; got $huber_delta."))
        new(smoothness_interannual, huber_delta)
    end
end

"""
    GP(; kernel=with_lengthscale(Matern52Kernel(), 1/6), obs_noise=1.0, n_quad=5, huber_delta=1.345)

Sparse inducing-point Gaussian Process (DTC approximation) on a monthly grid.
Supports arbitrary `KernelFunctions.jl` kernels (sums, products, periodic, etc.).

The `:std` layer in the returned `DimStack` is a spatially-varying sandwich standard
deviation: lower where observations are dense, higher where they are sparse.

# Keywords
- `kernel`: Any `KernelFunctions.jl` kernel. Default: Matérn-5/2 with 2-month lengthscale.
- `obs_noise::Float64 = 1.0`: Observation noise variance σ² **in the same units as y²**.
  Controls the GP posterior smoothness: smaller values → tighter fit to observations.
- `n_quad::Int = 5`: Gauss-Legendre quadrature points per interval.
- `huber_delta::Float64 = 1.345`: Threshold parameter for Huber loss (only used when
  `loss_norm = :Huber`). Controls transition from quadratic (L2) to linear (L1) behavior.
  Huber's recommendation: δ = 1.345 for 95% efficiency at the normal distribution.
"""
@kwdef struct GP{K} <: DisaggregationMethod
    kernel::K                        = with_lengthscale(Matern52Kernel(), 1/6)
    obs_noise::Float64               = 1.0
    n_quad::Int                      = 5
    huber_delta::Float64             = 1.345

    function GP(kernel::K, obs_noise, n_quad, huber_delta) where K
        huber_delta > 0 || throw(ArgumentError("huber_delta must be positive; got $huber_delta."))
        new{K}(kernel, obs_noise, n_quad, huber_delta)
    end
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


