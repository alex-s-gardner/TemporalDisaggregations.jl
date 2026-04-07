function disaggregate(m::GP,
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
    loss_norm ∈ (:L1, :L2, :Huber) ||
        throw(ArgumentError("loss_norm must be :L1, :L2, or :Huber; got :$loss_norm."))
    if !isnothing(weights)
        length(weights) == n ||
            throw(DimensionMismatch("weights must have the same length as aggregate_values."))
        all(>(0), weights) ||
            throw(ArgumentError("All weights must be positive."))
    end

    
    if !issorted(interval_start)
        order = sortperm(interval_start)
        t1    = yeardecimal.(interval_start[order])
        t2    = yeardecimal.(interval_end[order])
        y     = aggregate_values[order]
        w_obs = isnothing(weights) ? ones(n) : weights[order]
    else
        y = aggregate_values
        t1 = yeardecimal.(interval_start)
        t2 = yeardecimal.(interval_end)
        w_obs = isnothing(weights) ? ones(n) : weights
    end
    w_obs ./= mean(w_obs)


    # ── Inducing grid: 2× finer than output, floored at Day(1) ───────────────────
    inducing_period = _half_period(output_period)
    ind_start = minimum(interval_start)
    ind_end   = isnothing(output_end) ? maximum(interval_end) : output_end
    ind_dates, Z  = _date_grid(ind_start, ind_end, inducing_period)
    n_ind = length(Z)

    # ── Output grid ───────────────────────────────────────────────────────────────
    out_start = isnothing(output_start) ? ind_start : output_start
    out_dates, Z_out = _date_grid(out_start, ind_end, output_period)

    # ── Gauss-Legendre quadrature ─────────────────────────────────────────────
    # GL weights sum to 2 on [−1, 1]; dividing by 2 gives a mean-approximation
    # so that w' * f(tq) ≈ (1/Δt) ∫ f(t) dt.
    gl_nodes, gl_weights = gausslegendre(m.n_quad)
    w = gl_weights ./ 2

    # ── Integral cross-kernel C [n × m] ───────────────────────────────────────
    # C[i, k] = (1/Δtᵢ) ∫_{t1ᵢ}^{t2ᵢ} k(t, Z[k]) dt  ≈  w' K(tq_i, Z)
    C = Matrix{Float64}(undef, n, n_ind)
    Threads.@threads for i in 1:n
        mid    = (t1[i] + t2[i]) / 2
        half   = (t2[i] - t1[i]) / 2
        tq     = mid .+ half .* gl_nodes
        C[i, :] = kernelmatrix(m.kernel, tq, Z)' * w
    end

    # ── Kernel matrix at inducing points K [m × m] ───────────────────────────
    K_raw = kernelmatrix(m.kernel, Z, Z)
    jitter = 1e-6 * tr(K_raw) / n_ind
    K = Symmetric(K_raw + jitter * I)

    # ── Sparse GP posterior via matrix inversion lemma ────────────────────────
    # Σ_y = C K⁻¹ Cᵀ + σ²I  [n × n, never formed]
    # Woodbury: Σ_y⁻¹ = I/σ² − C M'⁻¹ Cᵀ / σ²   where M' = K σ² + CᵀWC

    W_obs  = Diagonal(w_obs)
    S_W    = C' * W_obs * C                  # [m × m]  (updated by IRLS for L1)
    M_W    = Symmetric(K .* σ² .+ S_W)       # [m × m]
    L_M    = cholesky(M_W)

    CtWy   = C' * W_obs * y                  # [m]
    v      = L_M \ CtWy                      # M'⁻¹ CᵀWy  [m]
    μ_Z    = (CtWy .- S_W * v) ./ σ²         # [m]

    # ── L1/Huber: IRLS refinement (skipped for :L2) ───────────────────────────
    # Replace W=W_obs with combined weights w_irls[i]*w_obs[i], iterate to convergence.
    # Predicted interval averages are C*μ_Z (not C*v, which is the Woodbury intermediate).
    if loss_norm == :L1 || loss_norm == :Huber
        ε_irls = 1e-6 * (std(y) + 1e-10)
        w_irls = ones(n)
        loss = _make_loss(loss_norm, m.huber_delta)
        for _ in 1:50
            W_eff      = Diagonal(w_irls .* w_obs)
            S_W        = C' * W_eff * C
            M_W        = Symmetric(K .* σ² .+ S_W)
            L_M        = cholesky(M_W)
            CtWy       = C' * W_eff * y
            v          = L_M \ CtWy
            μ_Z_new    = (CtWy .- S_W * v) ./ σ²
            r          = y .- C * v
            # Compute IRLS weights via LossFunctions.jl
            w_irls = _irls_weights(r, loss, ε_irls)
            _irls_converged(μ_Z_new, μ_Z) && (μ_Z = μ_Z_new; break)
            μ_Z = μ_Z_new
        end
    end

    # ── Kriging from inducing to output grid ──────────────────────────────────────
    K_out_Z = kernelmatrix(m.kernel, Z_out, Z)
    L_K     = cholesky(K)
    μ_out   = K_out_Z * (L_K \ μ_Z)

    r       = y .- C * v
    std_val = sqrt(sum(w_obs .* r.^2) / sum(w_obs))
    std_vec = fill(std_val, length(out_dates))

    return DimStack(
        (signal = DimArray(μ_out,    Ti(out_dates)),
         std    = DimArray(std_vec,  Ti(out_dates)));
        metadata = Dict(
            :method        => :gp,
            :kernel        => m.kernel,
            :obs_noise     => m.obs_noise,
            :n_quad        => m.n_quad,
            :loss_norm     => loss_norm,
            :output_period => output_period,
        )
    )
end
