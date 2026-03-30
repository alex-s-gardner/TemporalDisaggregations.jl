# Module-level solver — avoids a closure allocation on every disaggregate() call.
function _spline_solve(A, b)
    try
        return A \ b
    catch e
        e isa SingularException || rethrow()
        return pinv(A) * b
    end
end

function disaggregate(m::Spline,
                      aggregate_values::AbstractVector,
                      interval_start::AbstractVector{<:Dates.TimeType},
                      interval_end::AbstractVector{<:Dates.TimeType};
                      loss_norm::Symbol = :L2,
                      output_period::Dates.Period = Month(1),
                      output_start::Union{Dates.TimeType,Nothing} = nothing,
                      output_end::Union{Dates.TimeType,Nothing} = nothing,
                      weights::Union{AbstractVector,Nothing} = nothing)

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
    loss_norm ∈ (:L1, :L2) ||
        throw(ArgumentError("loss_norm must be :L1 or :L2; got :$loss_norm."))
    if !isnothing(weights)
        length(weights) == n ||
            throw(DimensionMismatch("weights must have the same length as aggregate_values."))
        all(>(0), weights) ||
            throw(ArgumentError("All weights must be positive."))
    end

    # Sort intervals chronologically
    order = sortperm(interval_start)
    t1    = yeardecimal.(interval_start[order])
    t2    = yeardecimal.(interval_end[order])
    y     = aggregate_values[order]
    w_obs = isnothing(weights) ? ones(n) : Float64.(weights[order])
    w_obs ./= mean(w_obs)

    # Quartic (p=4) B-spline space for F(t); x(t) = F′(t) is cubic.
    # Knot placement is the primary control over smoothness:
    #   monthly grid (default) → m ≈ 12·years, system is overdetermined, smooth by default.
    #   dense grid (n_knots=0) → m ≈ 2n, underdetermined, requires large smoothness.
    p_F        = 4
    t1_min     = minimum(t1)
    t2_max     = maximum(t2)
    t_range_yr = t2_max - t1_min
    t_nodes = if isnothing(m.n_knots)
        # Auto monthly: 12 knots per year, minimum p_F+2 to form a valid space
        n_auto = max(p_F + 2, round(Int, 12 * t_range_yr) + 1)
        collect(range(t1_min, t2_max; length = n_auto))
    elseif m.n_knots == 0
        # Dense: one knot per unique interval endpoint (old default; requires large λ)
        sort(unique(vcat(t1, t2)))
    else
        m.n_knots >= p_F + 1 ||
            throw(ArgumentError("n_knots must be ≥ $(p_F + 1) for degree-$p_F B-splines."))
        collect(range(t1_min, t2_max; length = m.n_knots))
    end
    k_F     = KnotVector(t_nodes) + p_F * KnotVector([t_nodes[1], t_nodes[end]])
    P_F     = BSplineSpace{p_F}(k_F)
    n_basis  = dim(P_F)
    if m.tension > 0.0
        n_basis >= 3 ||
            throw(ArgumentError(
                "tension > 0 requires at least 3 basis functions (n_basis=$n_basis); " *
                "increase n_knots or the time span."))
    end

    # Observation matrix C_norm: C_norm[i,j] = (B_j(t2[i]) − B_j(t1[i])) / Δt[i]
    # Built directly (avoids intermediate C matrix and a second n×n_basis allocation).
    Δt     = t2 .- t1
    C_norm = [(bsplinebasis(P_F, j, t2[i]) - bsplinebasis(P_F, j, t1[i])) / Δt[i]
              for i in 1:n, j in 1:n_basis]

    # P-spline penalty of order `penalty_order`: ‖Dᵣ a‖²
    Dr   = _difference_matrix(n_basis, m.penalty_order)
    DrDr = Dr' * Dr

    # Tension penalty: augments the P-spline with ‖D₂ a‖² ≈ ∫(x′(t))² dt,
    # scaled so tension=1 equates the two penalty magnitudes.
    # tension=0.0 → identical to existing P-spline (exact backward compatibility).
    if m.tension > 0.0
        D2   = _difference_matrix(n_basis, 2)
        D2D2 = D2' * D2
        scale = norm(DrDr) / (norm(D2D2) + 1e-10)
        P = DrDr + m.tension^2 * scale * D2D2
    else
        P = DrDr
    end

    # Weighted least-squares with P-spline (+ optional tension) regularisation.
    # λ scaled by ‖C'C‖/n so `smoothness` is dimensionless.
    # C_w = √w_obs ⊙ C_norm so that C_w'C_w = C_norm' Diag(w_obs) C_norm
    # via BLAS syrk (A'A path) — avoids the n_basis×n intermediate from C_norm'*Diag*C_norm.
    ε_irls    = 1e-6 * (std(y) + 1e-10)
    sqrt_w_obs = sqrt.(w_obs)
    C_w        = sqrt_w_obs .* C_norm
    CWC        = C_w' * C_w
    λ          = m.smoothness * (norm(CWC) / n + 1e-10)
    a          = _spline_solve(CWC + λ * P, C_norm' * (w_obs .* y))
    w_irls     = ones(n)                    # identity for L2; overwritten by L1
    if loss_norm == :L1
        for _ in 1:50
            r        = y .- C_norm * a
            @. w_irls = 1.0 / (abs(r) + ε_irls)
            w_eff_i  = w_irls .* w_obs
            C_w_i    = sqrt.(w_eff_i) .* C_norm
            CWC_e    = C_w_i' * C_w_i
            a_new    = _spline_solve(CWC_e + λ * P, C_norm' * (w_eff_i .* y))
            _irls_converged(a_new, a) && (a = a_new; break)
            a = a_new
        end
    end

    # Instantaneous signal: x(t) = F′(t) = Σⱼ aⱼ B′ⱼ(t)
    dP_F = BasicBSpline.derivative(P_F)

    # Evaluate on the output grid clamped to the data domain.
    # B_out [n_basis × n_out] is built once in column-major order (no transpose copy)
    # and reused for both the signal values and the sandwich variance.
    out_start = isnothing(output_start) ? minimum(interval_start) : output_start
    out_end   = isnothing(output_end)   ? maximum(interval_end)   : output_end
    out_dates, eval_times = _date_grid(out_start, out_end, output_period)
    eval_times = clamp.(eval_times, t_nodes[1], t_nodes[end])

    B_out  = [bsplinebasis(dP_F, j, t) for j in 1:n_basis, t in eval_times]  # [n_basis × n_out]
    values = vec(B_out' * a)                                                    # [n_out]

    r       = y .- C_norm * a
    std_val = sqrt(sum(w_obs .* r.^2) / sum(w_obs))

    # Sandwich variance: std varies with observation density.
    # f(t*) = b(t*)' M_eff⁻¹ C_norm' W_eff y  →  Var(f(t*)) = σ̂² v' (C_norm' W_eff C_norm) v
    # where v = M_eff⁻¹ b(t*).  We exploit C_norm' W_eff C_norm = M_eff − λP = CWC_eff to
    # compute q_vec[k] = (CWC_eff V[:,k]) · V[:,k] without forming the n×n_out matrix C_norm*V.
    w_eff    = w_irls .* w_obs
    C_we     = sqrt.(w_eff) .* C_norm
    CWC_eff  = C_we' * C_we                    # C_norm' Diag(w_eff) C_norm  [n_basis × n_basis]
    M_eff    = CWC_eff + λ * P
    V        = _spline_solve(M_eff, B_out)     # [n_basis × n_out]
    M_data_V = CWC_eff * V                     # [n_basis × n_out] — avoids n×n_out allocation
    q_vec    = max.(0.0, vec(sum(M_data_V .* V, dims=1)))  # clamp: PSD in theory, ±ε in practice
    q_mean   = mean(q_vec)
    std_vec  = q_mean > 1e-10 ? std_val .* sqrt.(q_vec ./ q_mean) : fill(std_val, length(q_vec))

    return DimStack(
        (signal = DimArray(values,  Ti(out_dates)),
         std    = DimArray(std_vec, Ti(out_dates)));
        metadata = Dict(
            :method        => :spline,
            :smoothness    => m.smoothness,
            :n_knots       => m.n_knots,
            :penalty_order => m.penalty_order,
            :tension       => m.tension,
            :loss_norm     => loss_norm,
            :output_period => output_period,
        )
    )
end
