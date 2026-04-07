function disaggregate(m::GP,
                      aggregate_values::AbstractVector,
                      interval_start::AbstractVector{<:Dates.TimeType},
                      interval_end::AbstractVector{<:Dates.TimeType};
                      loss_norm::DistanceLoss = L2DistLoss(),
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

    # Avoid Diagonal(w_obs) allocation: use broadcasting for row scaling
    C_W    = C .* w_obs                      # (n × m) .* (n,) broadcasts correctly
    S_W    = C_W' * C_W                      # syrk pattern [m × m]  (updated by IRLS for L1)
    M_W    = Symmetric(K .* σ² .+ S_W)       # [m × m]
    L_M    = cholesky(M_W)

    CtWy   = C' * (w_obs .* y)               # element-wise first [m]
    v      = L_M \ CtWy                      # M'⁻¹ CᵀWy  [m]
    μ_Z    = (CtWy .- S_W * v) ./ σ²         # [m]

    # ── L1/Huber: IRLS refinement (skipped for L2DistLoss) ────────────────────
    # Replace W=W_obs with combined weights w_irls[i]*w_obs[i], iterate to convergence.
    # Predicted interval averages are C*μ_Z (not C*v, which is the Woodbury intermediate).
    if !(loss_norm isa L2DistLoss)
        ε_irls = 1e-6 * (std(y) + 1e-10)
        w_irls = ones(n)
        for _ in 1:50
            w_eff      = w_irls .* w_obs             # combined weights (n,)
            C_W_eff    = C .* w_eff                  # row scaling (n × m) .* (n,)
            S_W        = C_W_eff' * C_W_eff
            M_W        = Symmetric(K .* σ² .+ S_W)
            L_M        = cholesky(M_W)
            CtWy       = C' * (w_eff .* y)
            v          = L_M \ CtWy
            μ_Z_new    = (CtWy .- S_W * v) ./ σ²
            r          = y .- C * v
            # Compute IRLS weights via LossFunctions.jl
            w_irls = _irls_weights(r, loss_norm, ε_irls)
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
            :loss_norm     => string(loss_norm),
            :output_period => output_period,
        )
    )
end

"""
    disaggregate(method::GP, Y::AbstractMatrix, interval_start, interval_end; kwargs...)

Batch variant of GP disaggregation for multiple time series sharing the same observation intervals.

Processes `n_series` columns of `Y` (each a length-`n` observation vector) and returns
`(signal, std, dates)` where `signal` and `std` are `n_eval × n_series` matrices.

The expensive cross-kernel `C` (computed via Gauss-Legendre quadrature) is shared across
all series, providing significant speedup compared to calling `disaggregate` in a loop.

For `:L2` loss, the entire solve is batched. For `:L1`/`:Huber`, IRLS is performed
per-series but the cross-kernel and inducing kernel are reused.

`weights` (if supplied) must be a length-`n` vector applied uniformly to all series.
"""
function disaggregate(m::GP,
                      Y::AbstractMatrix{<:Real},
                      interval_start::AbstractVector{<:Dates.TimeType},
                      interval_end::AbstractVector{<:Dates.TimeType};
                      loss_norm::DistanceLoss = L2DistLoss(),
                      output_period::Dates.Period = Month(1),
                      output_start::Union{Dates.TimeType,Nothing} = nothing,
                      output_end::Union{Dates.TimeType,Nothing} = nothing,
                      weights::Union{AbstractVector,Nothing} = nothing)

    σ²  = m.obs_noise
    n, n_series = size(Y)
    (length(interval_start) == n && length(interval_end) == n) ||
        throw(DimensionMismatch(
            "Y has $n rows but interval_start/interval_end have length $(length(interval_start))."))
    any(interval_end .<= interval_start) &&
        throw(ArgumentError("Every interval must satisfy interval_end > interval_start."))
    m.obs_noise >= 0 || throw(ArgumentError("obs_noise must be ≥ 0."))
    m.n_quad >= 3 || throw(ArgumentError("n_quad must be ≥ 3."))
    if !isnothing(weights)
        length(weights) == n ||
            throw(DimensionMismatch("weights must have length $n (one per observation row)."))
        all(>(0), weights) || throw(ArgumentError("All weights must be positive."))
    end

    # ── Sort by interval start ────────────────────────────────────────────────────
    if !issorted(interval_start)
        order = sortperm(interval_start)
        t1    = yeardecimal.(interval_start[order])
        t2    = yeardecimal.(interval_end[order])
        Y_ord = Y[order, :]
        w_obs = isnothing(weights) ? ones(n) : weights[order]
    else
        Y_ord = Y
        t1    = yeardecimal.(interval_start)
        t2    = yeardecimal.(interval_end)
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
    n_eval = length(out_dates)

    # ── Gauss-Legendre quadrature ─────────────────────────────────────────────────
    gl_nodes, gl_weights = gausslegendre(m.n_quad)
    w = gl_weights ./ 2

    # ── Integral cross-kernel C [n × m] (SHARED ACROSS ALL SERIES) ───────────────
    C = Matrix{Float64}(undef, n, n_ind)
    Threads.@threads for i in 1:n
        mid    = (t1[i] + t2[i]) / 2
        half   = (t2[i] - t1[i]) / 2
        tq     = mid .+ half .* gl_nodes
        C[i, :] = kernelmatrix(m.kernel, tq, Z)' * w
    end

    # ── Kernel matrix at inducing points K [m × m] (SHARED) ───────────────────────
    K_raw = kernelmatrix(m.kernel, Z, Z)
    jitter = 1e-6 * tr(K_raw) / n_ind
    K = Symmetric(K_raw + jitter * I)

    # ── Kriging matrix (SHARED) ───────────────────────────────────────────────────
    K_out_Z = kernelmatrix(m.kernel, Z_out, Z)
    L_K     = cholesky(K)

    # ── Allocate outputs ──────────────────────────────────────────────────────────
    Signal = Matrix{Float64}(undef, n_eval, n_series)
    Std    = Matrix{Float64}(undef, n_eval, n_series)

    if loss_norm isa L2DistLoss
        # ── Fully batched L2 path ─────────────────────────────────────────────────
        # Avoid Diagonal(w_obs) allocation: use broadcasting for row scaling
        C_W    = C .* w_obs                          # row scaling (n × m) .* (n,)
        S_W    = C_W' * C_W                          # [m × m]
        M_W    = Symmetric(K .* σ² .+ S_W)
        L_M    = cholesky(M_W)

        # Batch solve for all series at once
        CtWY   = C' * (Y_ord .* w_obs)               # [m × n_series]
        V      = L_M \ CtWY                           # [m × n_series]
        μ_Z    = (CtWY .- S_W * V) ./ σ²              # [m × n_series]

        # Batch kriging for all series
        Signal .= K_out_Z * (L_K \ μ_Z)               # [n_eval × n_series]

        # Per-series std (residual-based)
        w_sum = sum(w_obs)
        for j in 1:n_series
            r_j     = view(Y_ord, :, j) .- C * view(V, :, j)
            std_val = sqrt(dot(w_obs, r_j .^ 2) / w_sum)
            Std[:, j] .= std_val
        end

    else
        # ── Per-series IRLS, sharing C / K / K_out_Z ─────────────────────────────
        for j in 1:n_series
            y_j    = view(Y_ord, :, j)
            ε_irls = 1e-6 * (std(y_j) + 1e-10)
            w_irls = ones(n)
            μ_Z    = zeros(n_ind)  # initialize for convergence check
            v      = zeros(n_ind)  # Woodbury intermediate, needed for std computation

            # IRLS loop (init merged into first iteration)
            for _ in 1:50
                w_eff      = w_irls .* w_obs         # combined weights (n,)
                C_W_eff    = C .* w_eff              # row scaling (n × m) .* (n,)
                S_W        = C_W_eff' * C_W_eff
                M_W        = Symmetric(K .* σ² .+ S_W)
                L_M        = cholesky(M_W)
                CtWy       = C' * (w_eff .* y_j)
                v          = L_M \ CtWy
                μ_Z_new    = (CtWy .- S_W * v) ./ σ²
                r          = y_j .- C * v
                w_irls     = _irls_weights(r, loss_norm, ε_irls)
                _irls_converged(μ_Z_new, μ_Z) && (μ_Z = μ_Z_new; break)
                μ_Z = μ_Z_new
            end

            # Krig and store
            Signal[:, j] .= K_out_Z * (L_K \ μ_Z)

            # Per-series std
            r       = y_j .- C * v
            std_val = sqrt(sum((w_irls .* w_obs) .* r.^2) / sum(w_obs))
            Std[:, j] .= std_val
        end
    end

    return (signal = Signal, std = Std, dates = out_dates)
end
