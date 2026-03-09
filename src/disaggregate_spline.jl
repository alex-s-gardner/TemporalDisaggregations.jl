function disaggregate(m::Spline,
                      aggregate_values::AbstractVector,
                      interval_start::AbstractVector{<:Dates.TimeType},
                      interval_end::AbstractVector{<:Dates.TimeType};
                      loss_norm::Symbol = :L2,
                      output_period::Dates.Period = Month(1),
                      output_start::Union{Dates.TimeType,Nothing} = nothing)

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

    # Sort intervals chronologically
    order = sortperm(interval_start)
    t1    = yeardecimal.(interval_start[order])
    t2    = yeardecimal.(interval_end[order])
    y     = aggregate_values[order]

    # Quartic (p=4) B-spline space for F(t); x(t) = F′(t) is cubic.
    # Knot placement is the primary control over smoothness:
    #   monthly grid (default) → m ≈ 12·years, system is overdetermined, smooth by default.
    #   dense grid (n_knots=0) → m ≈ 2n, underdetermined, requires large smoothness.
    p_F        = 4
    t_range_yr = t2[end] - t1[1]
    t_nodes = if isnothing(m.n_knots)
        # Auto monthly: 12 knots per year, minimum p_F+2 to form a valid space
        n_auto = max(p_F + 2, round(Int, 12 * t_range_yr) + 1)
        collect(range(t1[1], t2[end]; length = n_auto))
    elseif m.n_knots == 0
        # Dense: one knot per unique interval endpoint (old default; requires large λ)
        sort(unique(vcat(t1, t2)))
    else
        m.n_knots >= p_F + 1 ||
            throw(ArgumentError("n_knots must be ≥ $(p_F + 1) for degree-$p_F B-splines."))
        collect(range(t1[1], t2[end]; length = m.n_knots))
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

    # Observation matrix C: C[i,j] = B_j(t2[i]) − B_j(t1[i])
    # By the fundamental theorem of calculus, C * a = areas  ⟺  F(t2ᵢ)−F(t1ᵢ) = areaᵢ
    C = [bsplinebasis(P_F, j, t2[i]) - bsplinebasis(P_F, j, t1[i])
         for i in 1:n, j in 1:n_basis]
    # Normalise each row by the interval length so we fit interval averages (y)
    # rather than interval totals (areas).  This gives equal weight to every
    # observation regardless of length and keeps RSS in signal² units.
    Δt     = Array(t2 .- t1)
    C_norm = C ./ Δt

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
    ε_irls = 1e-6 * (std(y) + 1e-10)
    CWC = C_norm' * C_norm
    λ   = m.smoothness * (norm(CWC) / n + 1e-10)
    a   = (CWC + λ * P) \ (C_norm' * y)          # L2 init (also final if loss_norm==:L2)
    if loss_norm == :L1
        w_irls = Vector{Float64}(undef, n)
        for _ in 1:50
            r     = y .- C_norm * a
            @. w_irls = 1.0 / (abs(r) + ε_irls)
            W_eff  = Diagonal(w_irls)
            CWC_e  = C_norm' * W_eff * C_norm
            a_new  = (CWC_e + λ * P) \ (C_norm' * W_eff * y)
            _irls_converged(a_new, a) && (a = a_new; break)
            a = a_new
        end
    end

    # Instantaneous signal: x(t) = F′(t) = Σⱼ aⱼ B′ⱼ(t)
    dP_F = BasicBSpline.derivative(P_F)

    # Evaluate on the output grid clamped to the data domain
    out_dates, eval_times = _date_grid(t_nodes[1], t_nodes[end], output_period; output_start)
    eval_times = clamp.(eval_times, t_nodes[1], t_nodes[end])

    values = [sum(a[j] * bsplinebasis(dP_F, j, t) for j in 1:n_basis) for t in eval_times]

    # Type-II MLE (empirical Bayes) for σ²:
    #   σ̂² = (‖y − C_norm â‖² + λ â′Pâ) / n
    # The second term is the marginal-likelihood penalty contribution, which
    # provides a finite noise floor when the system is underdetermined (m ≥ n).
    # This avoids the near-zero σ̂² that the naive rss/(n−df_fit) gives when
    # the spline can interpolate the data exactly.
    rss     = sum(abs2, C_norm * a .- y)                             # residuals in signal units
    σ̂²     = (rss + λ * dot(a, P * a)) / n
    # Small jitter ensures PD when λ is near zero (CWC rank-deficient for n_basis > n)
    A_unc   = Symmetric(CWC + λ * P)
    L_chol  = cholesky(A_unc + sqrt(eps()) * norm(A_unc) * I(n_basis))
    B_out   = Float64[bsplinebasis(dP_F, j, t) for t in eval_times, j in 1:n_basis]  # [n_out × n_basis]
    V_out   = L_chol.L \ B_out'                                      # [n_basis × n_out]
    std_vec = sqrt(σ̂²) .* sqrt.(dropdims(sum(abs2, V_out, dims=1), dims=1))

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
