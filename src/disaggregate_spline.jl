using BasicBSpline
using Dates
using Statistics
using LinearAlgebra

"""
    _difference_matrix(m, r)

Return the (m−r) × m matrix that computes r-th order differences of a length-m vector,
used as the P-spline smoothness penalty `‖Dᵣ a‖²`.  Built by composing the first-difference
operator r times, so each row contains the binomial coefficients of Δʳ with alternating signs.

| r | Approximates | Effect on fitted curve |
|---|---|---|
| 1 | ‖a‖ (constant penalty) | Shrinks toward zero |
| 2 | ∫ (x′′)² dt  | Penalises curvature |
| 3 | ∫ (x′′′)² dt | Penalises rate of curvature change |
| 4 | ∫ (x⁽⁴⁾)² dt | Very smooth; polynomial extrapolation |
"""
function _difference_matrix(m::Int, r::Int)
    r >= 1 || throw(ArgumentError("penalty_order must be ≥ 1"))
    r <= m - 1 || throw(ArgumentError("penalty_order must be < number of basis functions"))
    D = Matrix{Float64}(I, m, m)
    for _ in 1:r
        n  = size(D, 1) - 1
        Δ₁ = zeros(n, n + 1)
        for i in 1:n
            Δ₁[i, i]     = -1.0
            Δ₁[i, i + 1] =  1.0
        end
        D = Δ₁ * D
    end
    return D
end

"""
    _monthly_decimal_year_grid(t_min, t_max)

Return `(dates, decimal_years)` for a monthly `Date` grid covering `[t_min, t_max]`,
where both arguments are decimal years (e.g. `2020.5` = mid-2020).
"""
function _monthly_decimal_year_grid(t_min::Real, t_max::Real)
    y0 = floor(Int, t_min)
    m0 = clamp(floor(Int, (t_min - y0) * 12) + 1, 1, 12)
    y1 = floor(Int, t_max)
    m1 = clamp(floor(Int, (t_max - y1) * 12) + 1, 1, 12)
    dates = collect(Date(y0, m0, 1):Month(1):Date(y1, m1, 1))
    times = [year(d) + (month(d) - 1) / 12.0 for d in dates]
    return dates, times
end

"""
    disaggregate_spline(aggregate_values, interval_start, interval_end;
                   smoothness=1e-3, outlier_rejection=false,
                   n_knots=nothing, penalty_order=3, tension=0.0, loss_norm=:L2)

Reconstruct an instantaneous time series from interval-averaged observations.

Models the antiderivative F(t) of the instantaneous signal as a quartic (degree-4) B-spline
constrained so that F(t2ᵢ) − F(t1ᵢ) equals the observed area (average × duration) for each
interval. The instantaneous signal is recovered as x(t) = F′(t), which is cubic. This
formulation naturally handles overlapping, gapped, and out-of-order intervals.

# Arguments
- `aggregate_values`: Vector of n observed averages over each time interval.
- `interval_start`: Vector of interval start times as decimal years (e.g. `2023.0`).
- `interval_end`: Vector of interval end times as decimal years.
- `smoothness`: Non-negative regularization strength λ. Larger values produce smoother
  output at the cost of fidelity to the observations. Default `1e-3`.
- `outlier_rejection`: If `true`, intervals with |y − ȳ| > 2.5σ are down-weighted by ×0.1.
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

# Returns
Named tuple `(dates, values)` where `dates` is a `Vector{Date}` on a monthly grid spanning
the data domain and `values` is the reconstructed instantaneous signal at those dates.
"""
function disaggregate_spline(aggregate_values::AbstractVector,
                        interval_start::AbstractVector,
                        interval_end::AbstractVector;
                        smoothness::Real = 1e-3,
                        outlier_rejection::Bool = false,
                        n_knots::Union{Int, Nothing} = nothing,
                        penalty_order::Int = 3,
                        tension::Real = 0.0,
                        loss_norm::Symbol = :L2)

    n = length(aggregate_values)
    (length(interval_start) == n && length(interval_end) == n) ||
        throw(DimensionMismatch(
            "aggregate_values, interval_start, and interval_end must have equal length."))
    any(interval_end .<= interval_start) &&
        throw(ArgumentError(
            "Every interval must satisfy interval_end > interval_start."))
    tension >= 0 ||
        throw(ArgumentError("tension must be ≥ 0."))
    loss_norm ∈ (:L1, :L2) ||
        throw(ArgumentError("loss_norm must be :L1 or :L2; got :$loss_norm."))

    # Sort intervals chronologically
    order = sortperm(interval_start)
    t1    = Float64.(interval_start[order])
    t2    = Float64.(interval_end[order])
    y     = Float64.(aggregate_values[order])
    areas = y .* (t2 .- t1)

    # Observation weights (uniform unless outlier rejection is requested)
    w = ones(n)
    if outlier_rejection
        μ, σ = mean(y), std(y)
        for i in 1:n
            abs(y[i] - μ) > 2.5σ && (w[i] = 0.1)
        end
    end
    W = Diagonal(w)

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
    # λ scaled by ‖C'WC‖/n so `smoothness` is dimensionless.
    ε_irls = 1e-6 * (std(areas) + 1e-10)
    CWC = C' * W * C
    λ   = Float64(smoothness) * (norm(CWC) / n + 1e-10)
    a   = (CWC + λ * P) \ (C' * W * areas)          # L2 init (also final if loss_norm==:L2)
    if loss_norm == :L1
        for _ in 1:50
            r      = areas .- C * a
            w_irls = 1.0 ./ (abs.(r) .+ ε_irls)
            W_eff  = Diagonal(w .* w_irls)
            CWC_e  = C' * W_eff * C
            a_new  = (CWC_e + λ * P) \ (C' * W_eff * areas)
            maximum(abs.(a_new .- a)) / (norm(a) + 1e-10) < 1e-8 && (a = a_new; break)
            a = a_new
        end
    end

    # Instantaneous signal: x(t) = F′(t) = Σⱼ aⱼ B′ⱼ(t)
    dP_F = BasicBSpline.derivative(P_F)

    # Evaluate on a monthly grid clamped to the data domain
    monthly_dates, eval_times = _monthly_decimal_year_grid(t_nodes[1], t_nodes[end])
    eval_times = clamp.(eval_times, t_nodes[1], t_nodes[end])

    values = [sum(a[j] * bsplinebasis(dP_F, j, t) for j in 1:m) for t in eval_times]

    return (dates = monthly_dates, values = values)
end
