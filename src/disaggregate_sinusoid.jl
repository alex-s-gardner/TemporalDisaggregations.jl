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
                      loss_norm::DistanceLoss      = L2DistLoss(),
                      output_period::Dates.Period  = Month(1),
                      output_start::Union{Dates.TimeType,Nothing} = nothing,
                      output_end::Union{Dates.TimeType,Nothing} = nothing,
                      weights::Union{AbstractVector,Nothing} = nothing,
                      irls_tol::Float64 = 1e-8,
                      irls_max_iter::Int = 50)

    n = length(aggregate_values)
    (length(interval_start) == n && length(interval_end) == n) ||
        throw(DimensionMismatch(
            "aggregate_values, interval_start, and interval_end must have equal length."))
    any(interval_end .<= interval_start) &&
        throw(ArgumentError(
            "Every interval must satisfy interval_end > interval_start."))
    if !isnothing(weights)
        length(weights) == n ||
            throw(DimensionMismatch("weights must have the same length as aggregate_values."))
        all(>(0), weights) ||
            throw(ArgumentError("All weights must be positive."))
    end
    irls_tol > 0 ||
        throw(ArgumentError("irls_tol must be positive."))
    irls_max_iter >= 1 ||
        throw(ArgumentError("irls_max_iter must be >= 1."))

    # Sort chronologically
    order = sortperm(interval_start)
    t1    = yeardecimal.(interval_start[order])
    t2    = yeardecimal.(interval_end[order])
    y     = Float64.(aggregate_values[order])
    w_obs = isnothing(weights) ? ones(n) : Float64.(weights[order])
    w_obs ./= mean(w_obs)

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
    W_obs  = Diagonal(w_obs)
    DᵀW    = D' * W_obs
    θ      = (DᵀW * D + Λ) \ (DᵀW * y)          # L2 init
    w_irls = ones(n)                               # identity for L2; overwritten by L1/Huber
    if !(loss_norm isa L2DistLoss)
        for _ in 1:irls_max_iter
            r      = y .- D * θ
            # Compute IRLS weights via LossFunctions.jl
            w_irls = _irls_weights(r, loss_norm, ε_irls)
            W_eff  = Diagonal(w_irls .* w_obs)
            DᵀW_e  = D' * W_eff
            θ_new  = (DᵀW_e * D + Λ) \ (DᵀW_e * y)
            _irls_converged(θ_new, θ, irls_tol) && (θ = θ_new; break)
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
    out_start = isnothing(output_start) ? minimum(interval_start) : output_start
    out_end   = isnothing(output_end)   ? maximum(interval_end)   : output_end
    out_dates, eval_times = _date_grid(out_start, out_end, output_period)

    # Interpolate interannual anomalies with a natural cubic spline through year
    # centres (year + 0.5, γ_year), then add trend, offset, and seasonality.
    # This gives a smooth output free of year-boundary step discontinuities.
    γ_xs_sp = Float64.(years) .+ 0.5
    γ_ys_sp = [γ_dict[yr] for yr in years]
    n_γ     = length(γ_xs_sp)
    h_γ     = diff(γ_xs_sp)

    # Natural cubic spline helpers (used for signal and for sandwich variance below)
    function _γ_S(ys)   # second derivatives for natural spline through (γ_xs_sp, ys)
        n_γ > 2 || return zeros(n_γ)
        T   = Tridiagonal(h_γ[2:end-1],
                          [2(h_γ[i] + h_γ[i+1]) for i in 1:n_γ-2],
                          h_γ[2:end-1])
        rhs = [6 * ((ys[i+2] - ys[i+1]) / h_γ[i+1] -
                    (ys[i+1] - ys[i])   / h_γ[i])
               for i in 1:n_γ-2]
        [0.0; T \ rhs; 0.0]
    end
    function _γ_eval(t, ys, S)
        k = clamp(searchsortedlast(γ_xs_sp, t), 1, n_γ - 1)
        a = (γ_xs_sp[k+1] - t) / h_γ[k]
        b = (t - γ_xs_sp[k])   / h_γ[k]
        a * ys[k] + b * ys[k+1] + ((a^3 - a) * S[k] + (b^3 - b) * S[k+1]) * h_γ[k]^2 / 6
    end

    γ_S_fit = _γ_S(γ_ys_sp)
    values  = [μ_fit + β_fit * (t - t_ref) + _γ_eval(t, γ_ys_sp, γ_S_fit) +
               A_fit * sin(2π * t) + B_fit * cos(2π * t)
               for t in eval_times]

    r       = y .- D * θ
    std_val = sqrt(sum(w_obs .* r.^2) / sum(w_obs))

    # Sandwich variance: std varies with observation density.
    # f(t*) = e(t*)' θ,  Var(f(t*)) = σ̂² ‖e(t*)' G‖²
    # where G = Θ_inv D' √W_eff  and  Θ_inv = (D' W_eff D + Λ)⁻¹.
    n_out   = length(eval_times)
    w_eff   = w_irls .* w_obs
    Θ_mat   = D' * Diagonal(w_eff) * D + Λ
    G       = inv(Θ_mat) * (D' * Diagonal(sqrt.(w_eff)))     # [n_params × n]

    # Columns of E corresponding to γ_yr_k: evaluate spline with unit vector in pos k
    E_γ = zeros(n_out, n_years)
    for k in 1:n_years
        γ_unit    = zeros(n_years); γ_unit[k] = 1.0
        S_unit    = _γ_S(γ_unit)
        E_γ[:, k] = [_γ_eval(t, γ_unit, S_unit) for t in eval_times]
    end

    E   = hcat(ones(n_out), eval_times .- t_ref, E_γ,
               sin.(2π .* eval_times), cos.(2π .* eval_times))  # [n_out × n_params]
    EG  = E * G                                                   # [n_out × n]
    var_vec = std_val^2 .* vec(sum(EG.^2, dims=2))
    std_vec = sqrt.(var_vec)

    return DimStack(
        (signal = DimArray(values,  Ti(out_dates)),
         std    = DimArray(std_vec, Ti(out_dates)));
        metadata = Dict(
            :method                 => :sinusoid,
            :smoothness_interannual => m.smoothness_interannual,
            :loss_norm              => string(loss_norm),
            :output_period          => output_period,
            :mean                   => μ_fit,
            :trend                  => β_fit,
            :amplitude              => seasonal_amplitude,
            :phase                  => seasonal_phase,
            :interannual            => γ_dict,
        )
    )
end
