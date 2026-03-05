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

function disaggregate(m::Sinusoid,
                      aggregate_values::AbstractVector,
                      interval_start::AbstractVector{<:Dates.TimeType},
                      interval_end::AbstractVector{<:Dates.TimeType};
                      loss_norm::Symbol            = :L2,
                      output_period::Dates.Period  = Month(1),
                      output_start::Union{Date,Nothing} = nothing)

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
    t1    = decimal_year.(interval_start[order])
    t2    = decimal_year.(interval_end[order])
    y     = Float64.(aggregate_values[order])

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
    λ_vec[3:2+n_years] .= m.smoothness_interannual
    Λ                 = Diagonal(λ_vec)

    # ── Weighted least-squares solve ──────────────────────────────────────────
    ε_irls = 1e-6 * (std(y) + 1e-10)
    DᵀD = D' * D
    θ   = (DᵀD + Λ) \ (D' * y)                  # L2 init
    if loss_norm == :L1
        w_irls = Vector{Float64}(undef, n)
        for _ in 1:50
            r      = y .- D * θ
            @. w_irls = 1.0 / (abs(r) + ε_irls)
            W_eff  = Diagonal(w_irls)
            DᵀW_e  = D' * W_eff
            θ_new  = (DᵀW_e * D + Λ) \ (DᵀW_e * y)
            _irls_converged(θ_new, θ) && (θ = θ_new; break)
            θ = θ_new
        end
    end

    # ── Extract parameters ────────────────────────────────────────────────────
    μ_fit  = θ[1]
    β_fit  = θ[2]
    γ_dict = Dict(zip(years, θ[3:2+n_years]))
    A_fit  = θ[2 + n_years + 1]   # sin coefficient
    B_fit  = θ[2 + n_years + 2]   # cos coefficient

    seasonal_amplitude = sqrt(A_fit^2 + B_fit^2)
    # Peak of A·sin(2πt)+B·cos(2πt) is at t where d/dt = 0 → tan(2πt) = B/A
    seasonal_phase     = mod(atan(B_fit, A_fit) / (2π), 1.0)

    # ── Evaluate on output grid ───────────────────────────────────────────────
    out_dates, eval_times = _date_grid(t1[1], t2[end], output_period; output_start)

    # Interpolate interannual anomalies with a natural cubic spline through year
    # centres (year + 0.5, γ_year), then add trend, offset, and seasonality.
    # This gives a smooth output free of year-boundary step discontinuities.
    γ_xs_sp = Float64.(years) .+ 0.5
    γ_ys_sp = [γ_dict[yr] for yr in years]
    n_γ     = length(γ_xs_sp)
    h_γ     = diff(γ_xs_sp)

    # Second derivatives for the natural (zero-curvature) cubic spline
    γ_S = if n_γ > 2
        T   = Tridiagonal(h_γ[2:end-1],
                          [2(h_γ[i] + h_γ[i+1]) for i in 1:n_γ-2],
                          h_γ[2:end-1])
        rhs = [6 * ((γ_ys_sp[i+2] - γ_ys_sp[i+1]) / h_γ[i+1] -
                    (γ_ys_sp[i+1] - γ_ys_sp[i])   / h_γ[i])
               for i in 1:n_γ-2]
        [0.0; T \ rhs; 0.0]
    else
        zeros(n_γ)
    end

    function γ_interp(t)
        k = clamp(searchsortedlast(γ_xs_sp, t), 1, n_γ - 1)
        a = (γ_xs_sp[k+1] - t) / h_γ[k]
        b = (t - γ_xs_sp[k])   / h_γ[k]
        return a * γ_ys_sp[k] + b * γ_ys_sp[k+1] +
               ((a^3 - a) * γ_S[k] + (b^3 - b) * γ_S[k+1]) * h_γ[k]^2 / 6
    end

    values = [μ_fit + β_fit * (t - t_ref) + γ_interp(t) +
              A_fit * sin(2π * t) + B_fit * cos(2π * t)
              for t in eval_times]

    # WLS covariance propagation for uncertainty (L2 system; approximate for L1)
    L_chol  = cholesky(Symmetric(DᵀD + Λ))
    V_hat   = L_chol.L \ D'                                          # [n_params × n]
    df_fit  = sum(abs2, V_hat)
    rss     = sum(abs2, D * θ .- y)
    σ̂²     = rss / max(1.0, n - df_fit)
    D_out   = zeros(length(eval_times), n_params)
    for (i, t) in enumerate(eval_times)
        D_out[i, 1] = 1.0
        D_out[i, 2] = t - t_ref
        for (k, yr) in enumerate(years)
            D_out[i, 2 + k] = Float64(floor(Int, t) == yr)
        end
        D_out[i, 2 + n_years + 1] = sin(2π * t)
        D_out[i, 2 + n_years + 2] = cos(2π * t)
    end
    V_out   = L_chol.L \ D_out'                                      # [n_params × n_out]
    std_vec = sqrt(σ̂²) .* sqrt.(dropdims(sum(abs2, V_out, dims=1), dims=1))

    return DimStack(
        (signal = DimArray(values,  Ti(out_dates)),
         std    = DimArray(std_vec, Ti(out_dates)));
        metadata = Dict(
            :method                 => :sinusoid,
            :smoothness_interannual => m.smoothness_interannual,
            :loss_norm              => loss_norm,
            :output_period    => output_period,
            :mean                   => μ_fit,
            :trend                  => β_fit,
            :amplitude              => seasonal_amplitude,
            :phase                  => seasonal_phase,
            :interannual            => γ_dict,
        )
    )
end
