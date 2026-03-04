using LinearAlgebra
using Dates
using Statistics

"""
    _interval_sin_integral(t1, t2)

Return the average of sin(2πt) over [t1, t2]:
    (1/Δt) ∫_{t1}^{t2} sin(2πt) dt = (cos(2πt1) − cos(2πt2)) / (2π Δt)
"""
@inline function _interval_sin_integral(t1::Float64, t2::Float64)
    (cos(2π * t1) - cos(2π * t2)) / (2π * (t2 - t1))
end

"""
    _interval_cos_integral(t1, t2)

Return the average of cos(2πt) over [t1, t2]:
    (1/Δt) ∫_{t1}^{t2} cos(2πt) dt = (sin(2πt2) − sin(2πt1)) / (2π Δt)
"""
@inline function _interval_cos_integral(t1::Float64, t2::Float64)
    (sin(2π * t2) - sin(2π * t1)) / (2π * (t2 - t1))
end

"""
    _year_overlap_fraction(t1, t2, yr)

Fraction of the interval [t1, t2] that falls within calendar year `yr`
(i.e. the half-open interval [yr, yr+1)).
"""
@inline function _year_overlap_fraction(t1::Float64, t2::Float64, yr::Int)
    lo = max(t1, Float64(yr))
    hi = min(t2, Float64(yr + 1))
    max(0.0, hi - lo) / (t2 - t1)
end

"""
    disaggregate_sinusoid(aggregate_values, interval_start, interval_end;
                            smoothness_interannual = 1e-2,
                            obs_noise              = 1.0,
                            outlier_rejection      = false,
                            loss_norm              = :L2)

Reconstruct an instantaneous time series from interval-averaged observations by fitting
the parametric model

    x(t) = μ + β·(t − t̄) + γ(year(t)) + A·sin(2πt) + B·cos(2πt)

where
- `μ`         — overall mean,
- `β`         — linear trend (units per year),
- `γ(year)`   — inter-annual anomaly for each calendar year (one scalar per year),
- `A`, `B`    — cosine and sine amplitudes of the annual seasonal cycle;
                seasonal amplitude = √(A²+B²), peak time = atan(B,A)/(2π) yr.

All five components are fit **simultaneously** via weighted least squares. Because every
term integrates analytically over arbitrary intervals, no quadrature is required. The
design matrix is constructed in closed form:

    D[i, ·] = [1,  (t̄ᵢ−t̄),  overlap(i,yr₁)/Δtᵢ, …,  ⟨sin⟩ᵢ,  ⟨cos⟩ᵢ]

where t̄ᵢ = (t1ᵢ+t2ᵢ)/2 is the interval midpoint and ⟨·⟩ᵢ denotes the interval average.

The inter-annual anomalies are L2-regularised toward zero with strength
`smoothness_interannual`; set it higher to suppress year-to-year variation.

# Arguments
- `aggregate_values`: Vector of n observed averages over each time interval.
- `interval_start`, `interval_end`: Interval boundaries as decimal years.
- `smoothness_interannual`: Ridge penalty on inter-annual anomalies γ. Default `1e-2`.
- `obs_noise`: Observation noise variance σ² used for outlier down-weighting. Default `1.0`.
- `outlier_rejection`: If `true`, intervals with |y−ȳ| > 2.5σ are down-weighted by ×0.1.
- `loss_norm`: Loss function for the data-fit term. `:L2` (default) minimises the
  weighted sum of squared residuals. `:L1` uses Iteratively Reweighted Least Squares
  (IRLS) to minimise the sum of absolute residuals, which is more robust to outliers.

# Returns
Named tuple with fields:
- `dates`        — `Vector{Date}` on a monthly grid spanning the data domain.
- `values`       — Reconstructed instantaneous signal at `dates`.
- `mean`         — Fitted overall mean μ.
- `trend`        — Linear trend β (units/yr).
- `amplitude`    — Seasonal amplitude √(A²+B²).
- `phase`        — Fractional year of seasonal peak (e.g. 0.5 = mid-year).
- `interannual`  — `Dict{Int,Float64}` mapping calendar year → anomaly γ.
"""
function disaggregate_sinusoid(aggregate_values::AbstractVector,
                                 interval_start::AbstractVector,
                                 interval_end::AbstractVector;
                                 smoothness_interannual::Real = 1e-2,
                                 obs_noise::Real              = 1.0,
                                 outlier_rejection::Bool      = false,
                                 loss_norm::Symbol            = :L2)

    n = length(aggregate_values)
    (length(interval_start) == n && length(interval_end) == n) ||
        throw(DimensionMismatch(
            "aggregate_values, interval_start, and interval_end must have equal length."))
    any(interval_end .<= interval_start) &&
        throw(ArgumentError(
            "Every interval must satisfy interval_end > interval_start."))
    loss_norm ∈ (:L1, :L2) ||
        throw(ArgumentError("loss_norm must be :L1 or :L2; got :$loss_norm."))

    # Sort chronologically
    order = sortperm(interval_start)
    t1    = Float64.(interval_start[order])
    t2    = Float64.(interval_end[order])
    y     = Float64.(aggregate_values[order])

    # Observation weights
    w = ones(n)
    if outlier_rejection
        μ_y, σ_y = mean(y), std(y)
        for i in 1:n
            abs(y[i] - μ_y) > 2.5σ_y && (w[i] = 0.1)
        end
    end
    W = Diagonal(w)

    # ── Parameter layout ──────────────────────────────────────────────────────
    # θ = [μ,  β,  γ_yr1, …, γ_yrK,  A,  B]
    #      1   2   3 … K+2          K+3  K+4
    years       = minimum(floor.(Int, t1)):maximum(floor.(Int, t2))
    n_years     = length(years)
    n_params    = 2 + n_years + 2
    t_ref       = (t1[1] + t2[end]) / 2          # centering point for trend

    # ── Analytical design matrix ──────────────────────────────────────────────
    D = zeros(n, n_params)
    for i in 1:n
        # Mean
        D[i, 1] = 1.0
        # Trend (centred so mean and trend are less correlated)
        D[i, 2] = (t1[i] + t2[i]) / 2 - t_ref
        # Inter-annual: fraction of interval falling in each calendar year
        for (k, yr) in enumerate(years)
            D[i, 2 + k] = _year_overlap_fraction(t1[i], t2[i], yr)
        end
        # Annual seasonal sin and cos (exact integrals)
        D[i, 2 + n_years + 1] = _interval_sin_integral(t1[i], t2[i])
        D[i, 2 + n_years + 2] = _interval_cos_integral(t1[i], t2[i])
    end

    # ── L2 regularisation on inter-annual anomalies only ─────────────────────
    λ_vec             = zeros(n_params)
    λ_vec[3:2+n_years] .= Float64(smoothness_interannual)
    Λ                 = Diagonal(λ_vec)

    # ── Weighted least-squares solve ──────────────────────────────────────────
    ε_irls = 1e-6 * (std(y) + 1e-10)
    DᵀW = D' * W
    θ   = (DᵀW * D + Λ) \ (DᵀW * y)                # L2 init
    if loss_norm == :L1
        for _ in 1:50
            r      = y .- D * θ
            w_irls = 1.0 ./ (abs.(r) .+ ε_irls)
            W_eff  = Diagonal(w .* w_irls)
            DᵀW_e  = D' * W_eff
            θ_new  = (DᵀW_e * D + Λ) \ (DᵀW_e * y)
            maximum(abs.(θ_new .- θ)) / (norm(θ) + 1e-10) < 1e-8 && (θ = θ_new; break)
            θ = θ_new
        end
    end

    # ── Extract parameters ────────────────────────────────────────────────────
    μ_fit  = θ[1]
    β_fit  = θ[2]
    γ_dict = Dict(yr => θ[2 + k] for (k, yr) in enumerate(years))
    A_fit  = θ[2 + n_years + 1]   # sin coefficient
    B_fit  = θ[2 + n_years + 2]   # cos coefficient

    seasonal_amplitude = sqrt(A_fit^2 + B_fit^2)
    # Peak of A·sin(2πt)+B·cos(2πt) is at t where d/dt = 0 → tan(2πt) = B/A
    seasonal_phase     = mod(atan(B_fit, A_fit) / (2π), 1.0)

    # ── Evaluate on monthly grid ──────────────────────────────────────────────
    monthly_dates, eval_times = _monthly_decimal_year_grid(t1[1], t2[end])

    values = [(μ_fit + β_fit * (t - t_ref)
               + get(γ_dict, floor(Int, t), 0.0)
               + A_fit * sin(2π * t)
               + B_fit * cos(2π * t))
              for t in eval_times]

    return (
        dates        = monthly_dates,
        values       = values,
        mean         = μ_fit,
        trend        = β_fit,
        amplitude    = seasonal_amplitude,
        phase        = seasonal_phase,
        interannual  = γ_dict,
    )
end
