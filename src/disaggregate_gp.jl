function disaggregate(m::GP,
                      aggregate_values::AbstractVector,
                      interval_start::AbstractVector{<:Dates.TimeType},
                      interval_end::AbstractVector{<:Dates.TimeType};
                      loss_norm::Symbol = :L2,
                      output_period::Dates.Period = Month(1),
                      output_start::Union{Dates.TimeType,Nothing} = nothing,
                      output_end::Union{Dates.TimeType,Nothing} = nothing)

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

    order = sortperm(interval_start)
    t1    = yeardecimal.(interval_start[order])
    t2    = yeardecimal.(interval_end[order])
    y     = Array(aggregate_values[order])

    # ── Monthly inducing grid (always monthly for O(n_ind³) tractability) ──────
    monthly_dates, Z = _monthly_yeardecimal_grid(t1[1], t2[end])
    n_ind = length(Z)

    # ── Output grid (may differ from inducing grid) ────────────────────────────
    # Use inducing grid directly only when output is default monthly-on-1st.
    use_inducing_as_output = output_period == Month(1) && isnothing(output_start) && isnothing(output_end)
    t_out_end = isnothing(output_end) ? t2[end] : yeardecimal(output_end)
    out_dates, Z_out = use_inducing_as_output ?
        (monthly_dates, Z) : _date_grid(t1[1], t_out_end, output_period; output_start)

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
    K = Symmetric(kernelmatrix(m.kernel, Z, Z))

    # ── Sparse GP posterior via matrix inversion lemma ────────────────────────
    # Σ_y = C K⁻¹ Cᵀ + σ²I  [n × n, never formed]
    # Woodbury: Σ_y⁻¹ = I/σ² − C M'⁻¹ Cᵀ / σ²   where M' = K σ² + CᵀC

    S_W    = C' * C                           # [m × m]  (updated by IRLS for L1)
    M_W    = Symmetric(K .* σ² .+ S_W)       # [m × m]
    L_M    = cholesky(M_W)

    Cty    = C' * y                           # [m]
    v      = L_M \ Cty                        # M'⁻¹ Cᵀy  [m]
    μ_Z    = (Cty .- S_W * v) ./ σ²          # [m]

    # ── L1: IRLS refinement (skipped for :L2) ─────────────────────────────────
    # Replace W=I with per-obs weights w_irls[i] = 1/(|r_i|+ε), iterate to convergence.
    # Predicted interval averages are C*μ_Z (not C*v, which is the Woodbury intermediate).
    if loss_norm == :L1
        ε_irls = 1e-6 * (std(y) + 1e-10)
        w_irls = ones(n)
        for _ in 1:50
            W_eff      = Diagonal(w_irls)
            S_W        = C' * W_eff * C
            M_W        = Symmetric(K .* σ² .+ S_W)
            L_M        = cholesky(M_W)
            CtWy       = C' * W_eff * y
            v          = L_M \ CtWy
            μ_Z_new    = (CtWy .- S_W * v) ./ σ²
            r          = y .- C * μ_Z_new
            w_irls     = _irls_weights(r, ε_irls)
            _irls_converged(μ_Z_new, μ_Z) && (μ_Z = μ_Z_new; break)
            μ_Z = μ_Z_new
        end
    end

    # Posterior variance at inducing points Z (using final S_W and L_M):
    # Var[k] = K[k,k] − (S_W M_W'⁻¹ K)[k,k]
    # = K_diag − diag(S_W R)  where R = M_W'⁻¹ K
    R_mat    = L_M \ Matrix(K)                # M_W'⁻¹ K  [m × m]
    K_diag   = kernelmatrix_diag(m.kernel, Z)   # [m]
    post_var = K_diag .- dropdims(sum(S_W .* R_mat, dims=1), dims=1)

    if use_inducing_as_output
        return DimStack(
            (signal = DimArray(μ_Z,                        Ti(out_dates)),
             std    = DimArray(sqrt.(max.(0.0, post_var)), Ti(out_dates)));
            metadata = Dict(
                :method      => :gp,
                :kernel      => m.kernel,
                :obs_noise   => m.obs_noise,
                :n_quad      => m.n_quad,
                :loss_norm   => loss_norm,
                :output_period => output_period,
            )
        )
    else
        # Kriging prediction at arbitrary output grid via an extra O(m²) solve.
        # μ_out = K_out_Z K_ZZ⁻¹ μ_Z
        # Var_out[k] = K_out[k,k] − (S_W M_W'⁻¹ K_out^T)_columnsum[k]
        K_out_Z      = kernelmatrix(m.kernel, Z_out, Z)            # [n_out × m]
        L_K          = cholesky(K)                                # K_ZZ Cholesky
        μ_out        = K_out_Z * (L_K \ μ_Z)                    # [n_out]
        R_out        = L_M \ K_out_Z'                            # [m × n_out]
        K_diag_out   = kernelmatrix_diag(m.kernel, Z_out)
        # DTC predictive variance: K_diag_out - diag(K_out_Z * M_W^{-1} * K_Z_out)
        # diag(A B) = sum(A .* B', dims=2) for A [n×m] and B [m×n]
        post_var_out = K_diag_out .- dropdims(sum(K_out_Z .* R_out', dims=2), dims=2)
        return DimStack(
            (signal = DimArray(μ_out,                            Ti(out_dates)),
             std    = DimArray(sqrt.(max.(0.0, post_var_out)),   Ti(out_dates)));
            metadata = Dict(
                :method      => :gp,
                :kernel      => m.kernel,
                :obs_noise   => m.obs_noise,
                :n_quad      => m.n_quad,
                :loss_norm   => loss_norm,
                :output_period => output_period,
            )
        )
    end
end
