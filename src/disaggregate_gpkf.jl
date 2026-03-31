function disaggregate(m::GPKF,
                      aggregate_values::AbstractVector,
                      interval_start::AbstractVector{<:Dates.TimeType},
                      interval_end::AbstractVector{<:Dates.TimeType};
                      loss_norm::Symbol = :L2,
                      output_period::Dates.Period = Month(1),
                      output_start::Union{Dates.TimeType,Nothing} = nothing,
                      output_end::Union{Dates.TimeType,Nothing} = nothing,
                      weights::Union{AbstractVector,Nothing} = nothing)

    σ²  = m.obs_noise
    n   = length(aggregate_values)
    (length(interval_start) == n && length(interval_end) == n) ||
        throw(DimensionMismatch(
            "aggregate_values, interval_start, and interval_end must have equal length."))
    any(interval_end .<= interval_start) &&
        throw(ArgumentError(
            "Every interval must satisfy interval_end > interval_start."))
    m.obs_noise >= 0 ||
        throw(ArgumentError("obs_noise must be ≥ 0."))
    m.n_quad >= 3 || throw(ArgumentError("n_quad must be ≥ 3."))
    loss_norm ∈ (:L1, :L2) ||
        throw(ArgumentError("loss_norm must be :L1 or :L2; got :$loss_norm."))
    if !isnothing(weights)
        length(weights) == n ||
            throw(DimensionMismatch("weights must have the same length as aggregate_values."))
        all(>(0), weights) ||
            throw(ArgumentError("All weights must be positive."))
    end

    # ── Sort, convert to decimal years ───────────────────────────────────────
    order = sortperm(interval_start)
    t1    = yeardecimal.(interval_start[order])
    t2    = yeardecimal.(interval_end[order])
    y     = Array(aggregate_values[order])
    w_obs = isnothing(weights) ? ones(n) : Float64.(weights[order])
    w_obs ./= mean(w_obs)

    # ── GL quadrature nodes + weights normalized to sum to 1 ─────────────────
    gl_nodes, gl_weights = gausslegendre(m.n_quad)
    w_gl = gl_weights ./ 2              # sum(w_gl) == 1

    # ── Expand each interval into n_quad pseudo-point observations ────────────
    # Each pseudo-point j of interval i has:
    #   time:  t_ij = mid_i + half_i * gl_nodes[j]
    #   value: y[i]  (all pseudo-points carry the same observed average)
    #   noise: σ²_ij = σ² / (w_obs[i] * w_gl[j])
    # Total precision across n_quad points = sum_j (w_obs[i]*w_gl[j]/σ²)
    #                                      = w_obs[i]/σ² — matches original.
    N_all  = n * m.n_quad
    t_all  = Vector{Float64}(undef, N_all)
    y_all  = Vector{Float64}(undef, N_all)
    σ²_all = Vector{Float64}(undef, N_all)
    for i in 1:n
        mid  = (t1[i] + t2[i]) / 2
        half = (t2[i] - t1[i]) / 2
        for j in 1:m.n_quad
            idx         = (i - 1) * m.n_quad + j
            t_all[idx]  = mid + half * gl_nodes[j]
            y_all[idx]  = y[i]
            σ²_all[idx] = σ² / (w_obs[i] * w_gl[j])
        end
    end

    # ── Sort pseudo-points chronologically (required by Kalman filter) ────────
    perm      = sortperm(t_all)
    iperm     = invperm(perm)
    t_sorted  = t_all[perm]
    y_sorted  = y_all[perm]
    σ²_sorted = σ²_all[perm]

    # ── TemporalGPs Kalman filter (L2 first pass) ─────────────────────────────
    f_sde = to_sde(AbstractGPs.GP(m.kernel), SArrayStorage(Float64))
    fx    = f_sde(t_sorted, Diagonal(σ²_sorted))
    post  = posterior(fx, y_sorted)

    # ── Helper: posterior mean at pseudo-points → per-interval residuals ──────
    function _interval_residuals(p)
        μ_gl = mean(p(t_sorted, Diagonal(σ²_sorted)))  # posterior mean at training pts
        μ_u  = μ_gl[iperm]                              # restore original index order
        r_i  = Vector{Float64}(undef, n)
        for i in 1:n
            base     = (i - 1) * m.n_quad
            r_i[i]   = y[i] - dot(w_gl, view(μ_u, base+1:base+m.n_quad))
        end
        return r_i
    end

    r      = _interval_residuals(post)
    w_irls = ones(n)

    # ── L1: IRLS refinement ───────────────────────────────────────────────────
    if loss_norm == :L1
        ε_irls = 1e-6 * (std(y) + 1e-10)
        for _ in 1:50
            w_irls_new = _irls_weights(r, ε_irls)
            for i in 1:n, j in 1:m.n_quad
                σ²_all[(i-1)*m.n_quad + j] = σ² / (w_irls_new[i] * w_obs[i] * w_gl[j])
            end
            σ²_new  = σ²_all[perm]
            fx_new  = f_sde(t_sorted, Diagonal(σ²_new))
            post_new = posterior(fx_new, y_sorted)
            r_new    = _interval_residuals(post_new)
            _irls_converged(w_irls_new, w_irls) && (w_irls = w_irls_new; r = r_new; post = post_new; break)
            w_irls = w_irls_new; r = r_new; post = post_new
        end
    end

    # ── Output grid ───────────────────────────────────────────────────────────
    out_start = isnothing(output_start) ? minimum(interval_start) : output_start
    out_end   = isnothing(output_end)   ? maximum(interval_end)   : output_end
    out_dates, out_times = _date_grid(out_start, out_end, output_period)
    out_f64   = Float64.(out_times)
    n_out     = length(out_f64)

    # ── Posterior mean + variance on output grid ──────────────────────────────
    μ_out, v_out = mean_and_var(post(out_f64, Diagonal(fill(1e-9, n_out))))
    post_std     = sqrt.(max.(0.0, v_out .- 1e-9))

    # ── Sandwich std ─────────────────────────────────────────────────────────
    std_val    = sqrt(sum(w_obs .* r.^2) / sum(w_obs))
    mean_pstd  = mean(post_std)
    std_vec    = mean_pstd > 1e-10 ?
        std_val .* (post_std ./ mean_pstd) :
        fill(std_val, n_out)

    return DimStack(
        (signal = DimArray(μ_out,   Ti(out_dates)),
         std    = DimArray(std_vec, Ti(out_dates)));
        metadata = Dict(
            :method        => :gpkf,
            :kernel        => m.kernel,
            :obs_noise     => m.obs_noise,
            :n_quad        => m.n_quad,
            :loss_norm     => loss_norm,
            :output_period => output_period,
        )
    )
end
