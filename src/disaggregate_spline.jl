using BasicBSpline
using Dates
using Statistics
using LinearAlgebra

"""
    disaggregate_spline(aggregate_values, interval_start, interval_end;
                   smoothness=1e-3, n_knots=nothing, penalty_order=3,
                   tension=0.0, loss_norm=:L2)

Reconstruct an instantaneous time series from interval-averaged observations.

Models the antiderivative F(t) of the instantaneous signal as a quartic (degree-4) B-spline
constrained so that F(t2ᵢ) − F(t1ᵢ) equals the observed area (average × duration) for each
interval. The instantaneous signal is recovered as x(t) = F′(t), which is cubic. This
formulation naturally handles overlapping, gapped, and out-of-order intervals.

# Arguments
- `aggregate_values`: Vector of n observed averages over each time interval.
- `interval_start`: Vector of interval start times as `Date` or `DateTime` values.
- `interval_end`: Vector of interval end times as `Date` or `DateTime` values.
- `smoothness`: Non-negative regularization strength λ. Larger values produce smoother
  output at the cost of fidelity to the observations. Default `1e-3`.
- `n_knots`: Number of knots for the B-spline basis. `nothing` (default) automatically
  uses a **monthly grid** spanning the data domain (12 knots per year), giving a
  well-conditioned overdetermined system and smooth results. Pass an integer to override
  (e.g. `n_knots = 24` for bi-monthly resolution). Pass `0` to fall back to placing a knot
  at every unique interval endpoint (dense, high-fidelity but requires a much larger
  `smoothness` to avoid oscillation, and is slow for large n).
- `penalty_order`: Order r of the difference penalty ‖Dᵣ a‖². Default `3` (penalises
  rate of curvature change ≈ x′′′). Higher values give progressively smoother output and
  more natural boundary behaviour. Use `2` for a less aggressive penalty.
- `tension`: Non-negative tension strength τ ≥ 0. Adds a scaled first-derivative
  penalty `τ² · ‖D₂ a‖²` (≈ τ² ∫(x′(t))² dt) alongside the curvature penalty.
  The penalty is auto-scaled so `tension=1` gives equal weight to both terms.
  Default `0.0` reproduces the standard P-spline exactly.
  - `tension ≈ 0.5–1`  — moderate stiffening; suppresses oscillation
  - `tension ≈ 5–10`   — strong tension; approaches piecewise-linear behaviour
- `loss_norm`: Loss function for the data-fit term. `:L2` (default) minimises the
  weighted sum of squared residuals. `:L1` uses Iteratively Reweighted Least Squares
  (IRLS) to minimise the sum of absolute residuals, which is more robust to outliers.
- `output_step`: Temporal resolution of the output grid as a `Dates.Period`
  (e.g. `Day(1)`, `Week(1)`, `Month(3)`). Default `Month(1)`.

# Returns
`DimStack` with layers `values` and `std`, both `DimArray` with a `Ti(dates)` dimension.
`metadata(result)` returns `Dict(:method => :spline)`.
Uncertainty is a frequentist P-spline confidence band; approximate for `:L1` loss.
"""
function disaggregate_spline(aggregate_values::AbstractVector,
                        interval_start::AbstractVector{<:Dates.TimeType},
                        interval_end::AbstractVector{<:Dates.TimeType};
                        smoothness::Real = 1e-3,
                        n_knots::Union{Int, Nothing} = nothing,
                        penalty_order::Int = 3,
                        tension::Real = 0.0,
                        loss_norm::Symbol = :L2,
                        output_step::Dates.Period = Month(1))

    n = length(aggregate_values)
    (length(interval_start) == n && length(interval_end) == n) ||
        throw(DimensionMismatch(
            "aggregate_values, interval_start, and interval_end must have equal length."))
    any(interval_end .<= interval_start) &&
        throw(ArgumentError(
            "Every interval must satisfy interval_end > interval_start."))
    smoothness >= 0 ||
        throw(ArgumentError("smoothness must be ≥ 0."))
    tension >= 0 ||
        throw(ArgumentError("tension must be ≥ 0."))
    loss_norm ∈ (:L1, :L2) ||
        throw(ArgumentError("loss_norm must be :L1 or :L2; got :$loss_norm."))

    # Sort intervals chronologically
    order = sortperm(interval_start)
    t1    = _decimal_year.(interval_start[order])
    t2    = _decimal_year.(interval_end[order])
    y     = Float64.(aggregate_values[order])
    areas = y .* (t2 .- t1)

    # Quartic (p=4) B-spline space for F(t); x(t) = F′(t) is cubic.
    # Knot placement is the primary control over smoothness:
    #   monthly grid (default) → m ≈ 12·years, system is overdetermined, smooth by default.
    #   dense grid (n_knots=0) → m ≈ 2n, underdetermined, requires large smoothness.
    p_F        = 4
    t_range_yr = t2[end] - t1[1]
    t_nodes = if isnothing(n_knots)
        # Auto monthly: 12 knots per year, minimum p_F+2 to form a valid space
        n_auto = max(p_F + 2, round(Int, 12 * t_range_yr) + 1)
        collect(range(t1[1], t2[end]; length = n_auto))
    elseif n_knots == 0
        # Dense: one knot per unique interval endpoint (old default; requires large λ)
        sort(unique(vcat(t1, t2)))
    else
        n_knots >= p_F + 1 ||
            throw(ArgumentError("n_knots must be ≥ $(p_F + 1) for degree-$p_F B-splines."))
        collect(range(t1[1], t2[end]; length = n_knots))
    end
    k_F     = KnotVector(t_nodes) + p_F * KnotVector([t_nodes[1], t_nodes[end]])
    P_F     = BSplineSpace{p_F}(k_F)
    m       = dim(P_F)
    if tension > 0.0
        m >= 3 ||
            throw(ArgumentError(
                "tension > 0 requires at least 3 basis functions (m=$m); " *
                "increase n_knots or the time span."))
    end

    # Observation matrix C: C[i,j] = B_j(t2[i]) − B_j(t1[i])
    # By the fundamental theorem of calculus, C * a = areas  ⟺  F(t2ᵢ)−F(t1ᵢ) = areaᵢ
    C = [bsplinebasis(P_F, j, t2[i]) - bsplinebasis(P_F, j, t1[i])
         for i in 1:n, j in 1:m]

    # P-spline penalty of order `penalty_order`: ‖Dᵣ a‖²
    Dr   = _difference_matrix(m, penalty_order)
    DrDr = Dr' * Dr

    # Tension penalty: augments the P-spline with ‖D₂ a‖² ≈ ∫(x′(t))² dt,
    # scaled so tension=1 equates the two penalty magnitudes.
    # tension=0.0 → identical to existing P-spline (exact backward compatibility).
    if tension > 0.0
        D2   = _difference_matrix(m, 2)
        D2D2 = D2' * D2
        scale = norm(DrDr) / (norm(D2D2) + 1e-10)
        P = DrDr + Float64(tension)^2 * scale * D2D2
    else
        P = DrDr
    end

    # Weighted least-squares with P-spline (+ optional tension) regularisation.
    # λ scaled by ‖C'C‖/n so `smoothness` is dimensionless.
    ε_irls = 1e-6 * (std(areas) + 1e-10)
    CWC = C' * C
    λ   = Float64(smoothness) * (norm(CWC) / n + 1e-10)
    a   = (CWC + λ * P) \ (C' * areas)          # L2 init (also final if loss_norm==:L2)
    if loss_norm == :L1
        w_irls = Vector{Float64}(undef, n)
        for _ in 1:50
            r     = areas .- C * a
            @. w_irls = 1.0 / (abs(r) + ε_irls)
            W_eff  = Diagonal(w_irls)
            CWC_e  = C' * W_eff * C
            a_new  = (CWC_e + λ * P) \ (C' * W_eff * areas)
            _irls_converged(a_new, a) && (a = a_new; break)
            a = a_new
        end
    end

    # Instantaneous signal: x(t) = F′(t) = Σⱼ aⱼ B′ⱼ(t)
    dP_F = BasicBSpline.derivative(P_F)

    # Evaluate on the output grid clamped to the data domain
    out_dates, eval_times = _date_grid(t_nodes[1], t_nodes[end], output_step)
    eval_times = clamp.(eval_times, t_nodes[1], t_nodes[end])

    values = [sum(a[j] * bsplinebasis(dP_F, j, t) for j in 1:m) for t in eval_times]

    # Frequentist uncertainty via hat-matrix trace (L2 system; approximate for L1)
    # Small jitter ensures PD when λ is near zero (CWC rank-deficient for m > n)
    A_unc   = Symmetric(CWC + λ * P)
    L_chol  = cholesky(A_unc + sqrt(eps()) * norm(A_unc) * I(m))
    V_hat   = L_chol.L \ C'                                          # [m × n]
    df_fit  = sum(abs2, V_hat)
    rss     = sum(abs2, C * a .- areas)
    σ̂²     = rss / max(1.0, n - df_fit)
    B_out   = Float64[bsplinebasis(dP_F, j, t) for t in eval_times, j in 1:m]  # [n_out × m]
    V_out   = L_chol.L \ B_out'                                      # [m × n_out]
    std_vec = sqrt(σ̂²) .* sqrt.(dropdims(sum(abs2, V_out, dims=1), dims=1))

    return DimStack(
        (values = DimArray(values,  Ti(out_dates)),
         std    = DimArray(std_vec, Ti(out_dates)));
        metadata = Dict(
            :method        => :spline,
            :smoothness    => smoothness,
            :n_knots       => n_knots,
            :penalty_order => penalty_order,
            :tension       => tension,
            :loss_norm     => loss_norm,
        )
    )
end
