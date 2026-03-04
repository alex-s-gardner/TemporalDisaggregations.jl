using AbstractGPs
using KernelFunctions
using FastGaussQuadrature
using LinearAlgebra
using Dates
using Statistics

"""
    disaggregate_gp(aggregate_values, interval_start, interval_end;
                      kernel    = with_lengthscale(Matern52Kernel(), 1/6),
                      obs_noise = 1.0,
                      n_quad    = 5,
                      loss_norm = :L2)

Reconstruct an instantaneous time series from interval-averaged observations using a
sparse Gaussian Process (inducing-point / DTC approximation).

The instantaneous signal x(t) is modelled as a GP with the given `kernel`. Each
observation yᵢ is the temporal average of x(t) over [t1ᵢ, t2ᵢ]:

    yᵢ = (1/Δtᵢ) ∫_{t1ᵢ}^{t2ᵢ} x(t) dt + εᵢ,    εᵢ ~ N(0, σ²)

## Algorithm (sparse inducing-point GP)

A monthly grid Z of m ≈ 12·years inducing points is used. The integral of the kernel
over each observation interval is approximated with n_quad-point Gauss-Legendre
quadrature, producing an n × m cross-kernel matrix:

    C[i, k] = (1/Δtᵢ) ∫_{t1ᵢ}^{t2ᵢ} k(t, Z[k]) dt

The marginal observation covariance is Σ_y = C K⁻¹ Cᵀ + σ²I (n × n). Rather than
forming this explicitly, the matrix inversion lemma reduces the core solve to the
m × m system M' = K σ² + CᵀC:

    Σ_y⁻¹  =  I/σ²  −  C M'⁻¹ Cᵀ / σ²

The posterior mean and variance at the inducing grid then follow analytically:

    μ_Z    =  (Cᵀy − S v) / σ²             where  S = CᵀC,  v = M'⁻¹ Cᵀy
    Var[k] =  K[k,k] − [S M'⁻¹ K]_{k,k}

**Complexity:** O(n·m·n_quad) kernel evaluations + one O(m³) Cholesky of M'.
For a 5-year record (m ≈ 60), this is ~10⁵× faster than the O(n³) dense solve.

!!! note "Relationship to augmented state-space Kalman smoother"
    The true O(n) augmented state-space Kalman smoother (Tebbutt et al. 2021,
    https://proceedings.mlr.press/v161/tebbutt21a.html) would also be O(n) in memory
    and supports non-overlapping intervals with SDE-compatible kernels (Matérn 1/2,
    3/2, 5/2, Squared Exponential). This DTC inducing-point approach is used instead
    because it supports arbitrary kernels (sums, products, PeriodicKernel, etc.) and
    arbitrary overlapping intervals, with near-identical accuracy for monthly grids.

# Arguments
- `aggregate_values`: Vector of n observed interval averages.
- `interval_start`: Interval start times as `Date` or `DateTime` values.
- `interval_end`: Interval end times as `Date` or `DateTime` values.
- `kernel`: KernelFunctions.jl kernel for the GP prior on x(t). Sums and products of
  kernels are fully supported. Default: Matérn-5/2 with a 2-month length-scale.
- `obs_noise`: Observation noise variance σ² ≥ 0.
- `n_quad`: Gauss-Legendre quadrature points per interval (≥ 3). 5 is sufficient for
  sub-annual intervals; increase for intervals spanning several years.
- `loss_norm`: Loss function for the data-fit term. `:L2` (default) minimises the
  weighted sum of squared residuals. `:L1` uses Iteratively Reweighted Least Squares
  (IRLS) to minimise the sum of absolute residuals, which is more robust to outliers.
- `output_step`: Temporal resolution of the output grid as a `Dates.Period`
  (e.g. `Day(1)`, `Week(1)`, `Month(3)`). Default `Month(1)`. Inducing points are
  always kept on a monthly grid regardless of this setting; when `output_step ≠ Month(1)`
  the posterior is predicted at the output grid via an additional O(m²) kriging step.

# Returns
`DimStack` with layers `values` and `std`, both `DimArray` with a `Ti(dates)` dimension.
`metadata(result)` returns `Dict(:method => :gp)`.

# Example
```julia
k = 20^2 * with_lengthscale(Matern52Kernel(), 1/6)
result = disaggregate_gp(y, t1, t2; kernel = k, obs_noise = 4.0, n_quad = 5)
lines(result.dates, result.values)
band(result.dates, result.values .- 2result.std, result.values .+ 2result.std)
```
"""
function disaggregate_gp(aggregate_values::AbstractVector,
                            interval_start::AbstractVector{<:Dates.TimeType},
                            interval_end::AbstractVector{<:Dates.TimeType};
                            kernel    = with_lengthscale(Matern52Kernel(), 1/6),
                            obs_noise::Real = 1.0,
                            n_quad::Int = 5,
                            loss_norm::Symbol = :L2,
                            output_step::Dates.Period = Month(1))

    σ²  = Float64(obs_noise)
    n   = length(aggregate_values)
    (length(interval_start) == n && length(interval_end) == n) ||
        throw(DimensionMismatch(
            "aggregate_values, interval_start, and interval_end must have equal length."))
    any(interval_end .<= interval_start) &&
        throw(ArgumentError(
            "Every interval must satisfy interval_end > interval_start."))
    obs_noise >= 0 ||
        throw(ArgumentError("obs_noise must be ≥ 0."))
    n_quad >= 3 || throw(ArgumentError("n_quad must be ≥ 3."))
    loss_norm ∈ (:L1, :L2) ||
        throw(ArgumentError("loss_norm must be :L1 or :L2; got :$loss_norm."))

    order = sortperm(interval_start)
    t1    = _decimal_year.(interval_start[order])
    t2    = _decimal_year.(interval_end[order])
    y     = Float64.(aggregate_values[order])

    # ── Monthly inducing grid (always monthly for O(m³) tractability) ─────────
    monthly_dates, Z = _monthly_decimal_year_grid(t1[1], t2[end])
    m = length(Z)

    # ── Output grid (may differ from inducing grid) ────────────────────────────
    out_dates, Z_out = output_step == Month(1) ?
        (monthly_dates, Z) : _date_grid(t1[1], t2[end], output_step)

    # ── Gauss-Legendre quadrature ─────────────────────────────────────────────
    # GL weights sum to 2 on [−1, 1]; dividing by 2 gives a mean-approximation
    # so that w' * f(tq) ≈ (1/Δt) ∫ f(t) dt.
    gl_nodes, gl_weights = gausslegendre(n_quad)
    w = gl_weights ./ 2

    # ── Integral cross-kernel C [n × m] ───────────────────────────────────────
    # C[i, k] = (1/Δtᵢ) ∫_{t1ᵢ}^{t2ᵢ} k(t, Z[k]) dt  ≈  w' K(tq_i, Z)
    C = Matrix{Float64}(undef, n, m)
    Threads.@threads for i in 1:n
        mid    = (t1[i] + t2[i]) / 2
        half   = (t2[i] - t1[i]) / 2
        tq     = mid .+ half .* gl_nodes
        C[i, :] = kernelmatrix(kernel, tq, Z)' * w
    end

    # ── Kernel matrix at inducing points K [m × m] ───────────────────────────
    K = Symmetric(kernelmatrix(kernel, Z, Z))

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
    # Residuals use C*v (= C K⁻¹ μ_Z = predicted interval averages), not C*μ_Z.
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
            r          = y .- C * v
            w_irls_new = _irls_weights(r, ε_irls)
            _irls_converged(w_irls_new, w_irls) && (μ_Z = μ_Z_new; break)
            w_irls = w_irls_new
            μ_Z = μ_Z_new
        end
    end

    # Posterior variance at inducing points Z (using final S_W and L_M):
    # Var[k] = K[k,k] − (S_W M_W'⁻¹ K)[k,k]
    # = K_diag − diag(S_W R)  where R = M_W'⁻¹ K
    R_mat    = L_M \ Matrix(K)                # M_W'⁻¹ K  [m × m]
    K_diag   = kernelmatrix_diag(kernel, Z)   # [m]
    post_var = K_diag .- dropdims(sum(S_W .* R_mat, dims=1), dims=1)

    if output_step == Month(1)
        return DimStack(
            (values = DimArray(μ_Z,                        Ti(out_dates)),
             std    = DimArray(sqrt.(max.(0.0, post_var)), Ti(out_dates)));
            metadata = Dict(
                :method      => :gp,
                :kernel      => kernel,
                :obs_noise   => obs_noise,
                :n_quad      => n_quad,
                :loss_norm   => loss_norm,
            )
        )
    else
        # Kriging prediction at arbitrary output grid via an extra O(m²) solve.
        # μ_out = K_out_Z K_ZZ⁻¹ μ_Z
        # Var_out[k] = K_out[k,k] − (S_W M_W'⁻¹ K_out^T)_columnsum[k]
        K_out_Z      = kernelmatrix(kernel, Z_out, Z)            # [n_out × m]
        L_K          = cholesky(K)                                # K_ZZ Cholesky
        μ_out        = K_out_Z * (L_K \ μ_Z)                    # [n_out]
        R_out        = L_M \ K_out_Z'                            # [m × n_out]
        K_diag_out   = kernelmatrix_diag(kernel, Z_out)
        # DTC predictive variance: K_diag_out - diag(K_out_Z * M_W^{-1} * K_Z_out)
        # diag(A B) = sum(A .* B', dims=2) for A [n×m] and B [m×n]
        post_var_out = K_diag_out .- dropdims(sum(K_out_Z .* R_out', dims=2), dims=2)
        return DimStack(
            (values = DimArray(μ_out,                            Ti(out_dates)),
             std    = DimArray(sqrt.(max.(0.0, post_var_out)),   Ti(out_dates)));
            metadata = Dict(
                :method      => :gp,
                :kernel      => kernel,
                :obs_noise   => obs_noise,
                :n_quad      => n_quad,
                :loss_norm   => loss_norm,
            )
        )
    end
end
