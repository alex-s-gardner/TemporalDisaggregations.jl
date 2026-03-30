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
    loss_norm ∈ (:L1, :L2) ||
        throw(ArgumentError("loss_norm must be :L1 or :L2; got :$loss_norm."))
    if !isnothing(weights)
        length(weights) == n ||
            throw(DimensionMismatch("weights must have the same length as aggregate_values."))
        all(>(0), weights) ||
            throw(ArgumentError("All weights must be positive."))
    end

    order = sortperm(interval_start)
    t1    = yeardecimal.(interval_start[order])
    t2    = yeardecimal.(interval_end[order])
    y     = Array(aggregate_values[order])
    w_obs = isnothing(weights) ? ones(n) : Float64.(weights[order])
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

    # ── Integral cross-kernel C [n × n_ind] ───────────────────────────────────
    # C[i, k] = (1/Δtᵢ) ∫_{t1ᵢ}^{t2ᵢ} k(t, Z[k]) dt  ≈  w' K(tq_i, Z)
    C = Matrix{Float64}(undef, n, n_ind)
    Threads.@threads for i in 1:n
        mid    = (t1[i] + t2[i]) / 2
        half   = (t2[i] - t1[i]) / 2
        tq     = mid .+ half .* gl_nodes
        C[i, :] = kernelmatrix(m.kernel, tq, Z)' * w
    end

    # ── Kernel matrix at inducing points K [n_ind × n_ind] ───────────────────
    K_raw  = kernelmatrix(m.kernel, Z, Z)
    jitter = 1e-6 * tr(K_raw) / n_ind
    K_jit  = K_raw + jitter * I              # plain Matrix, includes jitter; reused for M_W
    K      = Symmetric(K_jit)

    # ── Sparse GP posterior via matrix inversion lemma ────────────────────────
    # Σ_y = C K⁻¹ Cᵀ + σ²I  [n × n, never formed]
    # Woodbury: Σ_y⁻¹ = I/σ² − C M'⁻¹ Cᵀ / σ²   where M' = K σ² + CᵀWC
    #
    # Row-scaled syrk: C_w = √w_obs ⊙ C → S_W = C_w'C_w = C' Diag(w_obs) C
    # avoids the n_ind×n intermediate from C' * Diagonal(w_obs) * C (~1.94 GB at n=1e5).
    sqrt_w_obs = sqrt.(w_obs)
    C_w        = sqrt_w_obs .* C              # [n × n_ind]
    S_W        = C_w' * C_w                  # [n_ind × n_ind] via BLAS syrk
    CtWy       = C_w' * (sqrt_w_obs .* y)    # [n_ind]

    # In-place M_W construction avoids K.*σ² temporary; M_W_buf reused in L1 loop.
    M_W_buf    = K_jit .* σ²
    M_W_buf  .+= S_W
    M_W        = Symmetric(M_W_buf)
    L_M        = cholesky(M_W)

    v      = L_M \ CtWy                      # M'⁻¹ CᵀWy  [n_ind]
    μ_Z    = (CtWy .- S_W * v) ./ σ²         # [n_ind]

    # ── L1: IRLS refinement (skipped for :L2) ─────────────────────────────────
    # Replace W=W_obs with combined weights w_irls[i]*w_obs[i], iterate to convergence.
    # Predicted interval averages are C*v (Woodbury intermediate), not C*μ_Z.
    # Pre-allocate C_w_eff and M_W_buf outside loop to avoid per-iteration allocation.
    w_irls = ones(n)                          # identity for L2; overwritten by L1
    if loss_norm == :L1
        ε_irls  = 1e-6 * (std(y) + 1e-10)
        C_w_eff = similar(C)                  # [n × n_ind], reused each iteration
        for _ in 1:50
            @. C_w_eff = sqrt(w_irls * w_obs) * C
            S_W        = C_w_eff' * C_w_eff   # syrk
            @. M_W_buf = K_jit * σ² + S_W
            M_W        = Symmetric(M_W_buf)
            L_M        = cholesky(M_W)
            CtWy       = C_w_eff' * (@. sqrt(w_irls * w_obs) * y)
            v          = L_M \ CtWy
            μ_Z_new    = (CtWy .- S_W * v) ./ σ²
            r          = y .- C * v
            w_irls     = _irls_weights(r, ε_irls)
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

    # Sandwich variance: std varies with observation density.
    # μ_out(t*) = k(t*,Z) M_W⁻¹ C' W_eff y  →  Var = σ̂² u' S_W u
    # where u = M_W⁻¹ k(Z,t*) and S_W = C' Diag(w_eff) C (already computed, consistent with L_M).
    # Identity: ‖√w_eff ⊙ C u‖² = u' S_W u = (S_W U)[:,k] · U[:,k]
    # Replaces n×n_out matrix C*U (~832 MB at n=1e5) with n_ind×n_out S_W*U (~20 MB).
    U       = L_M \ K_out_Z'                 # M_W⁻¹ K_out_Z^T  [n_ind × n_out]
    S_W_U   = S_W * U                        # [n_ind × n_out]
    var_vec = std_val^2 .* vec(sum(S_W_U .* U, dims=1))
    std_vec = sqrt.(max.(0.0, var_vec))

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
