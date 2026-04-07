# Module-level solver — avoids a closure allocation on every disaggregate() call.
function _spline_solve(A, b)
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
    loss_norm ∈ (:L1, :L2, :Huber) ||
        throw(ArgumentError("loss_norm must be :L1, :L2, or :Huber; got :$loss_norm."))
    if !isnothing(weights)
        length(weights) == n ||
            throw(DimensionMismatch("weights must have the same length as aggregate_values."))
        all(>(0), weights) ||
            throw(ArgumentError("All weights must be positive."))
    end

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
    # bsplinebasisall returns all p+1 non-zero basis values at a point in one de Boor pass,
    # replacing n_basis individual bsplinebasis calls (most of which returned 0).
    # Parallel :static schedule is safe: each thread writes only to its own row i.
    C_norm = zeros(Float64, n, n_basis)
    Threads.@threads for i in 1:n
        inv_dt = 1.0 / Δt[i]
        j1 = intervalindex(P_F, t1[i])
        j2 = intervalindex(P_F, t2[i])
        b1 = bsplinebasisall(P_F, j1, t1[i])
        b2 = bsplinebasisall(P_F, j2, t2[i])
        @inbounds for k in 0:p_F
            C_norm[i, j1 + k] -= b1[k+1] * inv_dt
            C_norm[i, j2 + k] += b2[k+1] * inv_dt
        end
    end

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
    ε_irls = 1e-6 * (std(aggregate_values) + 1e-10)
    C_w    = sqrt.(w_obs) .* C_norm
    CWC    = C_w' * C_w
    λ      = m.smoothness * (norm(CWC) / n + 1e-10)
    a      = _spline_solve(CWC + λ * P, C_norm' * (w_obs .* aggregate_values))
    w_irls = ones(n)                        # identity for L2; overwritten by L1/Huber

    # Pre-allocate residual; computed once for L2, updated each IRLS iteration for L1/Huber.
    r = Vector{Float64}(undef, n)
    if loss_norm == :L1 || loss_norm == :Huber
        # Pre-allocate IRLS buffers outside the loop — mirrors GP method pattern.
        loss = _make_loss(loss_norm, m.huber_delta)
        w_eff_i  = similar(w_obs)
        C_w_eff  = similar(C_norm)
        CWC_e    = Matrix{Float64}(undef, n_basis, n_basis)
        A_irls   = Matrix{Float64}(undef, n_basis, n_basis)
        w_eff_y  = similar(aggregate_values)
        rhs_irls = Vector{Float64}(undef, n_basis)
        mul!(r, C_norm, a); @. r = aggregate_values - r
        for _ in 1:50
            # Compute IRLS weights via LossFunctions.jl
            w_irls = _irls_weights(r, loss, ε_irls)
            @. w_eff_i = w_irls * w_obs
            @. C_w_eff = sqrt(w_eff_i) * C_norm   # row-broadcast: (n,) × (n,n_basis)
            mul!(CWC_e, C_w_eff', C_w_eff)
            @. A_irls  = CWC_e + λ * P
            @. w_eff_y = w_eff_i * aggregate_values
            mul!(rhs_irls, C_norm', w_eff_y)
            a_new = _spline_solve(A_irls, rhs_irls)
            mul!(r, C_norm, a_new); @. r = aggregate_values - r
            _irls_converged(a_new, a) && (a = a_new; break)
            a = a_new
        end
    else
        mul!(r, C_norm, a); @. r = aggregate_values - r
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

    # bsplinebasisall on the derivative space: one pass per output point instead of n_basis passes.
    n_eval = length(eval_times)
    B_out  = zeros(Float64, n_basis, n_eval)
    for (k, t) in enumerate(eval_times)
        j = intervalindex(P_F, t)
        b = bsplinebasisall(dP_F, j, t)
        @inbounds for l in 0:p_F
            B_out[j + l, k] = b[l+1]
        end
    end
    values = Vector{Float64}(undef, n_eval)
    mul!(values, B_out', a)

    std_val = sqrt(sum(w_obs .* r.^2) / sum(w_obs))

    # Sandwich variance: std varies with observation density.
    # f(t*) = b(t*)' M_eff⁻¹ C_norm' W_eff y  →  Var(f(t*)) = σ̂² v' (C_norm' W_eff C_norm) v
    # where v = M_eff⁻¹ b(t*).  We exploit C_norm' W_eff C_norm = M_eff − λP = CWC_eff to
    # compute q_vec[k] = (CWC_eff V[:,k]) · V[:,k] without forming the n×n_out matrix C_norm*V.
    # For L2, reuse C_w/CWC (w_irls=ones so w_eff=w_obs); for L1 use the IRLS final weights.
    if loss_norm == :L2
        CWC_eff = CWC
    else
        @. w_eff_i  = w_irls * w_obs
        @. C_w_eff  = sqrt(w_eff_i) * C_norm
        mul!(CWC_e, C_w_eff', C_w_eff)
        CWC_eff = CWC_e
    end
    M_eff    = CWC_eff + λ * P
    V        = _spline_solve(M_eff, B_out)     # [n_basis × n_out]
    M_data_V = CWC_eff * V                     # [n_basis × n_out] — avoids n×n_out allocation
    n_out_v  = size(V, 2)
    q_vec    = Vector{Float64}(undef, n_out_v)
    for k in 1:n_out_v                         # column-dot avoids n_basis×n_out temporary
        q_vec[k] = max(0.0, dot(view(M_data_V, :, k), view(V, :, k)))
    end
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

"""
    disaggregate(m::Spline, Y, interval_start, interval_end; kwargs...)

Batch variant: `Y` is an `n_obs × n_series` matrix where every column is an independent
time series sharing the same interval boundaries `interval_start` / `interval_end`.

Expensive structure — knot grid, observation matrix `C_norm`, penalty `P`, and the
Cholesky factorisation of the regularised Gram matrix — is computed **once** and reused
across all columns.  For `:L2` loss the entire solve is batched as three BLAS gemm calls
plus one multi-RHS triangular solve, giving near-linear scaling in `n_series`.

Returns a **named tuple** `(signal, std, dates)` where `signal` and `std` are
`n_eval × n_series` matrices and `dates` is the output date grid.

`weights` (if supplied) must be a length-`n_obs` vector applied uniformly to every series.
"""
function disaggregate(m::Spline,
                      Y::AbstractMatrix{<:Real},
                      interval_start::AbstractVector{<:Dates.TimeType},
                      interval_end::AbstractVector{<:Dates.TimeType};
                      loss_norm::Symbol = :L2,
                      output_period::Dates.Period = Month(1),
                      output_start::Union{Dates.TimeType,Nothing} = nothing,
                      output_end::Union{Dates.TimeType,Nothing} = nothing,
                      weights::Union{AbstractVector,Nothing} = nothing)

    n, n_series = size(Y)
    (length(interval_start) == n && length(interval_end) == n) ||
        throw(DimensionMismatch(
            "Y has $n rows but interval_start/interval_end have length $(length(interval_start))."))
    any(interval_end .<= interval_start) &&
        throw(ArgumentError("Every interval must satisfy interval_end > interval_start."))
    m.smoothness >= 0 || throw(ArgumentError("smoothness must be ≥ 0."))
    m.tension    >= 0 || throw(ArgumentError("tension must be ≥ 0."))
    loss_norm ∈ (:L1, :L2, :Huber) ||
        throw(ArgumentError("loss_norm must be :L1, :L2, or :Huber; got :$loss_norm."))
    if !isnothing(weights)
        length(weights) == n ||
            throw(DimensionMismatch("weights must have length $n (one per observation row)."))
        all(>(0), weights) || throw(ArgumentError("All weights must be positive."))
    end

    # ── Shared setup (identical to single-series path) ────────────────────────
    order = sortperm(interval_start)
    t1    = yeardecimal.(interval_start[order])
    t2    = yeardecimal.(interval_end[order])
    w_obs = isnothing(weights) ? ones(n) : Float64.(weights[order])
    w_obs ./= mean(w_obs)

    p_F        = 4
    t1_min     = minimum(t1);  t2_max = maximum(t2)
    t_range_yr = t2_max - t1_min
    t_nodes = if isnothing(m.n_knots)
        n_auto = max(p_F + 2, round(Int, 12 * t_range_yr) + 1)
        collect(range(t1_min, t2_max; length = n_auto))
    elseif m.n_knots == 0
        sort(unique(vcat(t1, t2)))
    else
        m.n_knots >= p_F + 1 ||
            throw(ArgumentError("n_knots must be ≥ $(p_F + 1) for degree-$p_F B-splines."))
        collect(range(t1_min, t2_max; length = m.n_knots))
    end
    k_F    = KnotVector(t_nodes) + p_F * KnotVector([t_nodes[1], t_nodes[end]])
    P_F    = BSplineSpace{p_F}(k_F)
    n_basis = dim(P_F)

    Δt     = t2 .- t1
    C_norm = zeros(Float64, n, n_basis)
    Threads.@threads for i in 1:n
        inv_dt = 1.0 / Δt[i]
        j1 = intervalindex(P_F, t1[i]);  j2 = intervalindex(P_F, t2[i])
        b1 = bsplinebasisall(P_F, j1, t1[i])
        b2 = bsplinebasisall(P_F, j2, t2[i])
        @inbounds for k in 0:p_F
            C_norm[i, j1 + k] -= b1[k+1] * inv_dt
            C_norm[i, j2 + k] += b2[k+1] * inv_dt
        end
    end

    Dr   = _difference_matrix(n_basis, m.penalty_order)
    DrDr = Dr' * Dr
    P = if m.tension > 0.0
        D2   = _difference_matrix(n_basis, 2)
        D2D2 = D2' * D2
        DrDr + m.tension^2 * (norm(DrDr) / (norm(D2D2) + 1e-10)) * D2D2
    else
        DrDr
    end

    C_w = sqrt.(w_obs) .* C_norm          # n × n_basis
    CWC = C_w' * C_w                       # n_basis × n_basis — shared Gram matrix
    λ   = m.smoothness * (norm(CWC) / n + 1e-10)
    M   = Symmetric(CWC + λ * P)
    # Factor once; reused for every column (L2) or as warm-start (L1).
    M_fact = cholesky(M; check = false)
    if !issuccess(M_fact)
        M_fact = nothing   # fall back to _spline_solve per column
    end
    _batch_solve(F, RHS) = isnothing(F) ? _spline_solve(M, RHS) : F \ Matrix(RHS)

    # Output grid — shared
    out_start = isnothing(output_start) ? minimum(interval_start) : output_start
    out_end   = isnothing(output_end)   ? maximum(interval_end)   : output_end
    out_dates, eval_times = _date_grid(out_start, out_end, output_period)
    eval_times = clamp.(eval_times, t_nodes[1], t_nodes[end])
    dP_F  = BasicBSpline.derivative(P_F)
    n_eval = length(eval_times)
    B_out  = zeros(Float64, n_basis, n_eval)
    for (k, t) in enumerate(eval_times)
        j = intervalindex(P_F, t)
        b = bsplinebasisall(dP_F, j, t)
        @inbounds for l in 0:p_F; B_out[j + l, k] = b[l+1]; end
    end

    # ── Reorder Y rows once ────────────────────────────────────────────────────
    Y_ord = Y[order, :]                    # n × n_series

    # ── Allocate outputs ───────────────────────────────────────────────────────
    Signal = Matrix{Float64}(undef, n_eval, n_series)
    Std    = Matrix{Float64}(undef, n_eval, n_series)

    if loss_norm == :L2
        # ── Fully batched L2 path ─────────────────────────────────────────────
        # RHS = C_norm' * Diag(w_obs) * Y_ord  [n_basis × n_series]
        WY  = w_obs .* Y_ord               # broadcast: (n,1) .* (n, n_series)
        RHS = C_norm' * WY                 # one gemm
        A   = _batch_solve(M_fact, RHS)    # one multi-RHS triangular solve
        mul!(Signal, B_out', A)            # one gemm: n_eval × n_series
        # Residuals = Y_ord - C_norm * A   [n × n_series]
        Res = Y_ord - C_norm * A           # one gemm
        # Sandwich variance — q_vec is data-independent, compute once
        V        = _batch_solve(M_fact, B_out)   # n_basis × n_eval
        M_data_V = CWC * V
        q_vec    = [max(0.0, dot(view(M_data_V, :, k), view(V, :, k))) for k in 1:n_eval]
        q_mean   = mean(q_vec)
        q_scale  = q_mean > 1e-10 ? sqrt.(q_vec ./ q_mean) : ones(n_eval)
        w_sum    = sum(w_obs)
        for j in 1:n_series
            r_j     = view(Res, :, j)
            std_val = sqrt(dot(w_obs, r_j .^ 2) / w_sum)
            Std[:, j] = std_val .* q_scale
        end

    else  # :L1 or :Huber
        # ── Per-series IRLS, sharing C_norm / P / B_out ───────────────────────
        # Allocate IRLS buffers once, reuse across series
        w_eff_i  = Vector{Float64}(undef, n)
        C_w_eff  = Matrix{Float64}(undef, n, n_basis)
        CWC_e    = Matrix{Float64}(undef, n_basis, n_basis)
        A_irls   = Matrix{Float64}(undef, n_basis, n_basis)
        rhs_irls = Vector{Float64}(undef, n_basis)
        r        = Vector{Float64}(undef, n)
        w_irls   = ones(n)
        loss = _make_loss(loss_norm, m.huber_delta)
        for j in 1:n_series
            y_j    = view(Y_ord, :, j)
            ε_irls = 1e-6 * (std(y_j) + 1e-10)
            # warm-start from L2 solution
            rhs_j  = C_norm' * (w_obs .* y_j)
            a      = _batch_solve(M_fact, rhs_j)
            fill!(w_irls, 1.0)
            mul!(r, C_norm, a); @. r = y_j - r
            for _ in 1:50
                # Compute IRLS weights via LossFunctions.jl
                w_irls = _irls_weights(r, loss, ε_irls)
                @. w_eff_i = w_irls * w_obs
                @. C_w_eff = sqrt(w_eff_i) * C_norm
                mul!(CWC_e, C_w_eff', C_w_eff)
                @. A_irls  = CWC_e + λ * P
                @. w_eff_i = w_eff_i * y_j   # reuse as w_eff_y
                mul!(rhs_irls, C_norm', w_eff_i)
                a_new = _spline_solve(A_irls, rhs_irls)
                mul!(r, C_norm, a_new); @. r = y_j - r
                _irls_converged(a_new, a) && (a = a_new; break)
                a = a_new
            end
            mul!(view(Signal, :, j), B_out', a)
            # Per-series sandwich variance (CWC_eff depends on final w_irls)
            @. w_eff_i = w_irls * w_obs     # restore (was overwritten above)
            @. C_w_eff = sqrt(w_eff_i) * C_norm
            mul!(CWC_e, C_w_eff', C_w_eff)
            M_eff_j  = CWC_e + λ * P
            V_j      = _spline_solve(M_eff_j, B_out)
            Mdv_j    = CWC_e * V_j
            q_vec_j  = [max(0.0, dot(view(Mdv_j, :, k), view(V_j, :, k))) for k in 1:n_eval]
            q_mean_j = mean(q_vec_j)
            w_sum    = sum(w_obs)
            std_val  = sqrt(dot(w_irls .* w_obs, r .^ 2) / w_sum)
            Std[:, j] = q_mean_j > 1e-10 ?
                std_val .* sqrt.(q_vec_j ./ q_mean_j) : fill(std_val, n_eval)
        end
    end

    return (signal = Signal, std = Std, dates = out_dates)
end
