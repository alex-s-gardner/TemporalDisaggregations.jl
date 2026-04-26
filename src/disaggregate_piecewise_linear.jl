# Helper functions for piecewise linear (hat function) basis
# Hat function: φₖ(t) = max(0, 1 - |t - tₖ| / h)
# Support: [tₖ - h, tₖ + h]

"""
    _period_to_years(p::Dates.Period) -> Float64

Convert a `Dates.Period` to decimal years for automatic knot spacing calculations.
Used to determine knot density based on the user-requested `output_period`.
"""
function _period_to_years(p::Dates.Period)::Float64
    p isa Year  && return Float64(Dates.value(p))
    p isa Month && return Float64(Dates.value(p)) / 12.0
    p isa Week  && return Float64(Dates.value(p)) * 7.0 / 365.25
    p isa Day   && return Float64(Dates.value(p)) / 365.25
    p isa Hour  && return Float64(Dates.value(p)) / (365.25 * 24)
    # Fallback for sub-hourly periods
    return Float64(Dates.value(Dates.Millisecond(p))) / (365.25 * 24 * 3600 * 1000)
end

"""
    _hat_antiderivative(t::Float64, t_knot::Float64, h::Float64) -> Float64

Compute the antiderivative Φₖ(t) of the hat function φₖ(t) = max(0, 1 - |t - tₖ| / h).

The antiderivative is piecewise quadratic:
- 0                              if t < tₖ - h
- (t - (tₖ-h))² / (2h)           if tₖ - h ≤ t < tₖ
- h/2 + (t-tₖ) - (t-tₖ)²/(2h)    if tₖ ≤ t < tₖ + h
- h                               if t ≥ tₖ + h
"""
function _hat_antiderivative(t::Float64, t_knot::Float64, h::Float64)::Float64
    Δ = t - t_knot
    if Δ < -h
        return 0.0
    elseif Δ < 0.0
        # Left side: (t - (t_knot - h))^2 / (2h) = (Δ + h)^2 / (2h)
        return (Δ + h)^2 / (2h)
    elseif Δ < h
        # Right side: h/2 + Δ - Δ^2/(2h)
        return 0.5h + Δ - Δ^2 / (2h)
    else
        return h
    end
end

"""
    _hat_integral(a::Float64, b::Float64, t_knot::Float64, h::Float64) -> Float64

Compute ∫[a,b] φₖ(t) dt where φₖ(t) is a hat function centered at `t_knot` with spacing `h`.
Returns Φₖ(b) - Φₖ(a) using the analytical antiderivative.
"""
function _hat_integral(a::Float64, b::Float64, t_knot::Float64, h::Float64)::Float64
    return _hat_antiderivative(b, t_knot, h) - _hat_antiderivative(a, t_knot, h)
end

"""
    _hat_value(t::Float64, t_knot::Float64, h::Float64) -> Float64

Evaluate hat function φₖ(t) = max(0, 1 - |t - tₖ| / h) at point t.
"""
function _hat_value(t::Float64, t_knot::Float64, h::Float64)::Float64
    Δ = abs(t - t_knot)
    return Δ < h ? 1.0 - Δ / h : 0.0
end

# Module-level solver — avoids a closure allocation on every disaggregate() call.
function _pwl_solve(A, b)
    try
        return A \ b
    catch e
        e isa SingularException || rethrow()
        try
            return pinv(A) * b
        catch e2
            e2 isa LAPACKException || rethrow()
            # Matrix is numerically catastrophic (NaN/Inf); return zeros.
            return zeros(eltype(b), size(A, 2), size(b)[2:end]...)
        end
    end
end

function disaggregate(m::PiecewiseLinear,
                      aggregate_values::AbstractVector,
                      interval_start::AbstractVector{<:Dates.TimeType},
                      interval_end::AbstractVector{<:Dates.TimeType};
                      loss_norm::DistanceLoss = L2DistLoss(),
                      output_period::Dates.Period = Month(1),
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
    m.smoothness >= 0 ||
        throw(ArgumentError("smoothness must be ≥ 0."))
    m.tension >= 0 ||
        throw(ArgumentError("tension must be ≥ 0."))
    m.n_knots >= 0 ||
        throw(ArgumentError("n_knots must be ≥ 0 (0 = auto)."))
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

    # Sort intervals chronologically
    if !issorted(interval_start)
        order = sortperm(interval_start)
        t1    = yeardecimal.(interval_start[order])
        t2    = yeardecimal.(interval_end[order])
        aggregate_values     = aggregate_values[order]
        w_obs = isnothing(weights) ? ones(n) : weights[order]
    else
        t1 = yeardecimal.(interval_start)
        t2 = yeardecimal.(interval_end)
        w_obs = isnothing(weights) ? ones(n) : weights
    end

    w_obs ./= mean(w_obs)

    # Knot grid setup
    t1_min     = minimum(t1)
    t2_max     = maximum(t2)
    t_range_yr = t2_max - t1_min

    if m.n_knots > 0
        n_knots = m.n_knots
    else
        # Auto: approximately one knot per output_period
        period_years = _period_to_years(output_period)
        n_knots = max(3, round(Int, t_range_yr / period_years) + 1)
    end

    n_knots >= 3 ||
        throw(ArgumentError(
            "n_knots must be >= 3 for meaningful first-order penalty (got $n_knots). " *
            "Increase n_knots or the time span."))

    if m.tension > 0.0 && n_knots < 3
        throw(ArgumentError(
            "tension > 0 requires at least 3 basis functions (n_knots=$n_knots)."))
    end

    t_knots = collect(range(t1_min, t2_max; length = n_knots))
    h = n_knots > 1 ? (t2_max - t1_min) / (n_knots - 1) : 1.0

    # Observation matrix C: C[i,k] = (1/Δt[i]) * ∫[t1[i], t2[i]] φₖ(t) dt
    # Only knots within [t1[i] - h, t2[i] + h] contribute (sparse rows).
    Δt = t2 .- t1
    C = zeros(Float64, n, n_knots)

    Threads.@threads for i in 1:n
        inv_dt = 1.0 / Δt[i]
        # Only non-zero for knots within extended interval
        k_min = searchsortedfirst(t_knots, t1[i] - h)
        k_max = searchsortedlast(t_knots, t2[i] + h)
        k_min = max(1, k_min)
        k_max = min(n_knots, k_max)

        @inbounds for k in k_min:k_max
            integral = _hat_integral(t1[i], t2[i], t_knots[k], h)
            C[i, k] = integral * inv_dt
        end
    end

    # First-order difference penalty: ‖D₁ θ‖²
    D1   = _difference_matrix(n_knots, 1)
    D1D1 = D1' * D1

    # Optional tension penalty: blend toward pure interpolation
    # tension = 0.0 → standard first-order penalty
    # tension = 1.0 → ridge penalty (penalize ‖θ‖²)
    if m.tension > 0.0
        scale = norm(D1D1) / (n_knots + 1e-10)
        P = (1.0 - m.tension) * D1D1 + m.tension * scale * I
    else
        P = D1D1
    end

    # Weighted least-squares with first-order regularization.
    # λ scaled by ‖C'C‖/n so `smoothness` is dimensionless.
    # C_w = √w_obs ⊙ C so that C_w'C_w = C' Diag(w_obs) C
    ε_irls = 1e-6 * (std(aggregate_values) + 1e-10)
    C_w    = sqrt.(w_obs) .* C
    CWC    = C_w' * C_w
    λ      = m.smoothness * (norm(CWC) / n + 1e-10)
    θ      = _pwl_solve(CWC + λ * P, C' * (w_obs .* aggregate_values))
    w_irls = ones(n)  # identity for L2; overwritten by L1/Huber

    # Pre-allocate residual; computed once for L2, updated each IRLS iteration for L1/Huber.
    r = Vector{Float64}(undef, n)
    if !(loss_norm isa L2DistLoss)
        # Pre-allocate IRLS buffers outside the loop — mirrors Spline method pattern.
        w_eff_i  = similar(w_obs)
        C_w_eff  = similar(C)
        CWC_e    = Matrix{Float64}(undef, n_knots, n_knots)
        A_irls   = Matrix{Float64}(undef, n_knots, n_knots)
        w_eff_y  = similar(aggregate_values)
        rhs_irls = Vector{Float64}(undef, n_knots)
        mul!(r, C, θ); @. r = aggregate_values - r
        for _ in 1:irls_max_iter
            # Compute IRLS weights via LossFunctions.jl
            w_irls = _irls_weights(r, loss_norm, ε_irls)
            @. w_eff_i = w_irls * w_obs
            @. C_w_eff = sqrt(w_eff_i) * C   # row-broadcast: (n,) × (n,n_knots)
            mul!(CWC_e, C_w_eff', C_w_eff)
            @. A_irls  = CWC_e + λ * P
            @. w_eff_y = w_eff_i * aggregate_values
            mul!(rhs_irls, C', w_eff_y)
            θ_new = _pwl_solve(A_irls, rhs_irls)
            mul!(r, C, θ_new); @. r = aggregate_values - r
            _irls_converged(θ_new, θ, irls_tol) && (θ = θ_new; break)
            θ = θ_new
        end
    else
        mul!(r, C, θ); @. r = aggregate_values - r
    end

    # Evaluate on the output grid clamped to the data domain.
    # B_out [n_knots × n_eval] is built once in column-major order (no transpose copy)
    # and reused for both the signal values and the sandwich variance.
    out_start = isnothing(output_start) ? minimum(interval_start) : output_start
    out_end   = isnothing(output_end)   ? maximum(interval_end)   : output_end
    out_dates, eval_times = _date_grid(out_start, out_end, output_period)
    eval_times = clamp.(eval_times, t1_min, t2_max)

    n_eval = length(eval_times)
    B_out  = zeros(Float64, n_knots, n_eval)
    for (j, t) in enumerate(eval_times)
        @inbounds for k in 1:n_knots
            B_out[k, j] = _hat_value(t, t_knots[k], h)
        end
    end
    values = Vector{Float64}(undef, n_eval)
    mul!(values, B_out', θ)

    std_val = sqrt(sum(w_obs .* r.^2) / sum(w_obs))

    # Sandwich variance: std varies with observation density.
    # f(t*) = b(t*)' M_eff⁻¹ C' W_eff y  →  Var(f(t*)) = σ̂² v' (C' W_eff C) v
    # where v = M_eff⁻¹ b(t*).  We exploit C' W_eff C = M_eff − λP = CWC_eff to
    # compute q_vec[k] = (CWC_eff V[:,k]) · V[:,k] without forming the n×n_eval matrix C*V.
    # For L2, reuse C_w/CWC (w_irls=ones so w_eff=w_obs); for L1 use the IRLS final weights.
    if loss_norm isa L2DistLoss
        CWC_eff = CWC
    else
        @. w_eff_i  = w_irls * w_obs
        @. C_w_eff  = sqrt(w_eff_i) * C
        mul!(CWC_e, C_w_eff', C_w_eff)
        CWC_eff = CWC_e
    end
    M_eff    = CWC_eff + λ * P
    V        = _pwl_solve(M_eff, B_out)     # [n_knots × n_eval]
    M_data_V = CWC_eff * V                   # [n_knots × n_eval] — avoids n×n_eval allocation
    n_out_v  = size(V, 2)
    q_vec    = Vector{Float64}(undef, n_out_v)
    for k in 1:n_out_v                       # column-dot avoids n_knots×n_eval temporary
        q_vec[k] = max(0.0, dot(view(M_data_V, :, k), view(V, :, k)))
    end
    q_mean   = mean(q_vec)
    std_vec  = q_mean > 1e-10 ? std_val .* sqrt.(q_vec ./ q_mean) : fill(std_val, length(q_vec))

    return DimStack(
        (signal = DimArray(values,  Ti(out_dates)),
         std    = DimArray(std_vec, Ti(out_dates)));
        metadata = Dict(
            :method        => :piecewise_linear,
            :smoothness    => m.smoothness,
            :n_knots       => n_knots,
            :tension       => m.tension,
            :loss_norm     => string(loss_norm),
            :output_period => output_period,
        )
    )
end
