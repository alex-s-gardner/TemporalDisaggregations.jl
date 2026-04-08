function disaggregate(m::GP,
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

    # Additional jitter for M_W matrix (K*σ² + S_W) to ensure positive definiteness
    # Use more aggressive jitter with a minimum floor value
    jitter_M = max(1e-4, 1e-5 * (tr(K_raw) * σ² / n_ind))

    # ── Sparse GP posterior via matrix inversion lemma ────────────────────────
    # Σ_y = C K⁻¹ Cᵀ + σ²I  [n × n, never formed]
    # Woodbury: Σ_y⁻¹ = I/σ² − C M'⁻¹ Cᵀ / σ²   where M' = K σ² + CᵀWC

    W_obs  = Diagonal(w_obs)
    S_W    = C' * W_obs * C                  # [m × m]  (updated by IRLS for L1)

    # Adaptive jitter: progressively increase if cholesky fails
    L_M = nothing
    jitter_adaptive = jitter_M
    for attempt in 1:10
        try
            M_W = Symmetric(K .* σ² .+ S_W + jitter_adaptive * I)
            L_M = cholesky(M_W)
            break
        catch e
            if e isa LinearAlgebra.PosDefException && attempt < 10
                jitter_adaptive *= 10.0  # Increase jitter by 10x and retry
            else
                rethrow()
            end
        end
    end

    CtWy   = C' * W_obs * y                  # [m]
    v      = L_M \ CtWy                      # M'⁻¹ CᵀWy  [m]
    μ_Z    = (CtWy .- S_W * v) ./ σ²         # [m]

    # ── L1/Huber: IRLS refinement (skipped for L2DistLoss) ────────────────────
    # Replace W=W_obs with combined weights w_irls[i]*w_obs[i], iterate to convergence.
    # Predicted interval averages are C*μ_Z (not C*v, which is the Woodbury intermediate).
    if !(loss_norm isa L2DistLoss)
        ε_irls = 1e-6 * (std(y) + 1e-10)
        # Compute initial residuals from L2 solution
        r          = y .- C * μ_Z
        # Compute initial IRLS weights based on L2 residuals
        w_irls = _irls_weights(r, loss_norm, ε_irls)

        for _ in 1:irls_max_iter
            W_eff      = Diagonal(w_irls .* w_obs)
            S_W        = C' * W_eff * C

            # Adaptive jitter for IRLS iterations
            jitter_adaptive = jitter_M
            for attempt in 1:10
                try
                    M_W = Symmetric(K .* σ² .+ S_W + jitter_adaptive * I)
                    L_M = cholesky(M_W)
                    break
                catch e
                    if e isa LinearAlgebra.PosDefException && attempt < 10
                        jitter_adaptive *= 10.0
                    else
                        rethrow()
                    end
                end
            end
            CtWy       = C' * W_eff * y
            v          = L_M \ CtWy
            μ_Z_new    = (CtWy .- S_W * v) ./ σ²
            r          = y .- C * μ_Z_new
            # Compute IRLS weights via LossFunctions.jl
            w_irls_new = _irls_weights(r, loss_norm, ε_irls)
            _irls_converged(μ_Z_new, μ_Z, irls_tol) && (μ_Z = μ_Z_new; break)
            μ_Z = μ_Z_new
            w_irls = w_irls_new
        end
    end

    # ── Kriging from inducing to output grid ──────────────────────────────────────
    K_out_Z = kernelmatrix(m.kernel, Z_out, Z)
    L_K     = cholesky(K)
    μ_out   = K_out_Z * (L_K \ μ_Z)

    # ── Sandwich standard error ───────────────────────────────────────────────────
    # Following Spline method: use data Gram matrix S_W in quadratic form, not full M_W
    # std(t*) = σ̂ · √(q(t*) / q̄) where q(t*) = V[:,t*]' · S_W · V[:,t*]
    # and V = M_W^(-1) · K_out_Z'
    # Compute residuals from kriged output (not inducing points)
    # The issue is that C * μ_Z uses the inducing point approximation, which can have
    # large residuals even when the final kriged output fits well. We need to use the
    # actual predicted interval averages from the output signal.
    r_avg = TemporalDisaggregations.interval_average(
        DimStack((signal=DimArray(μ_out, Ti(out_dates)),); metadata=Dict()),
        interval_start, interval_end)
    r = aggregate_values .- r_avg
    std_val = sqrt(sum(w_obs .* r.^2) / sum(w_obs))

    # V = M_W^(-1) · K_out_Z'  [n_ind × n_out]
    V = L_M \ K_out_Z'  # reuse L_M from IRLS/L2 solve

    # Use final IRLS weights for robust losses
    if loss_norm isa L2DistLoss
        S_W_eff = S_W
    else
        W_eff = Diagonal(w_irls .* w_obs)
        S_W_eff = C' * W_eff * C  # [n_ind × n_ind]
    end

    # M_data_V = S_W_eff · V (data contribution to variance)
    M_data_V = S_W_eff * V  # [n_ind × n_out]

    # Compute spatial density factor q_vec
    n_out = length(out_dates)
    q_vec = Vector{Float64}(undef, n_out)
    for k in 1:n_out
        q_vec[k] = max(0.0, dot(view(M_data_V, :, k), view(V, :, k)))
    end

    # Normalize by mean and scale by residual RMS (Spline approach)
    q_mean  = mean(q_vec)
    std_vec = q_mean > 1e-10 ? std_val .* sqrt.(q_vec ./ q_mean) : fill(std_val, n_out)

    return DimStack(
        (signal = DimArray(μ_out,    Ti(out_dates)),
         std    = DimArray(std_vec,  Ti(out_dates)));
        metadata = Dict(
            :method        => :gp,
            :kernel        => m.kernel,
            :obs_noise     => m.obs_noise,
            :n_quad        => m.n_quad,
            :loss_norm     => string(loss_norm),
            :output_period => output_period,
        )
    )
end
