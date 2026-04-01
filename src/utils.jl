using Dates
using LinearAlgebra

"""
    _date_grid(start::Dates.TimeType, stop::Dates.TimeType, step::Dates.Period)

Return `(dates, yeardecimals)` for a grid `start:step:stop` where `start` and `stop`
are `Date` or `DateTime`. Sub-daily steps coerce `Date` inputs to `DateTime`.
"""
function _date_grid(start::Dates.TimeType, stop::Dates.TimeType, step::Dates.Period)
    dates = collect(start:step:stop)
    decyr = yeardecimal.(dates)
    return dates, decyr
end

"""
    _half_period(p::Dates.Period) -> Dates.Period

Return a period approximately half the size of `p`, for use as the GP inducing
grid spacing (2× the output resolution). Sub-daily periods are floored at `Day(1)`
to keep the O(m³) Cholesky tractable.

| Input          | Output       |
|----------------|--------------|
| Year(n)        | Month(6n)    |
| Month(n), n≥2  | Month(n÷2)   |
| Month(1)       | Week(2)      |
| Week(n), n≥2   | Day(7n÷2)    |
| Week(1)        | Day(4)       |
| Day(n), n≥2    | Day(n÷2)     |
| Day(1) or finer| Day(1)       |
"""
function _half_period(p::Dates.Period)::Dates.Period
    v = Dates.value(p)
    p isa Year   && return Month(6v)
    p isa Month  && return v >= 2 ? Month(v ÷ 2) : Week(2)
    p isa Week   && return v >= 2 ? Day(max(1, 7v ÷ 2)) : Day(4)
    p isa Day    && return v >= 2 ? Day(v ÷ 2) : Day(1)
    return Day(1)  # sub-daily floor
end

# ─────────────────────────────────────────────────────────────────────────────
# B-spline / P-spline helpers
# ─────────────────────────────────────────────────────────────────────────────

"""
    _difference_matrix(m, r)

Return the (m−r) × m matrix that computes r-th order differences of a length-m vector,
used as the P-spline smoothness penalty `‖Dᵣ a‖²`. Built directly from binomial
coefficients with alternating signs, avoiding the O(r) matrix-multiplication chain.

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
    rows = m - r
    D = zeros(Float64, rows, m)
    for i in 1:rows
        for j in 0:r
            D[i, i + j] = Float64((-1)^(r - j) * binomial(r, j))
        end
    end
    return D
end

# ─────────────────────────────────────────────────────────────────────────────
# IRLS helpers
# ─────────────────────────────────────────────────────────────────────────────

"""
    _irls_weights(residuals, ε)

Return per-observation IRLS weights `1 / (|r| + ε)`.
"""
_irls_weights(r::AbstractVector, ε::Float64) = @. 1.0 / (abs(r) + ε)

"""
    _irls_converged(x_new, x, tol=1e-8)

Return `true` when the relative change `‖x_new − x‖∞ / (‖x‖ + 1e-10) < tol`.
"""
_irls_converged(x_new, x, tol=1e-8) =
    maximum(abs.(x_new .- x)) / (norm(x) + 1e-10) < tol

# ─────────────────────────────────────────────────────────────────────────────
# Post-processing helpers
# ─────────────────────────────────────────────────────────────────────────────

"""
    interval_average(result, t1, t2) → Vector{Float64}

Compute the time-average of a disaggregated signal over each observation interval
`[t1[i], t2[i]]` using trapezoidal integration on the high-resolution output grid.

# Arguments
- `result`: return value of `disaggregate` (has `.signal` field with a `:Ti` dimension)
- `t1`, `t2`: vectors of interval start/end times (DateTime or any type accepted by
  `DateFormats.yeardecimal`)

# Returns
`Vector{Float64}` of length `length(t1)`, each entry being the mean signal value over
the corresponding interval.  Intervals that fall entirely outside the output grid are
filled with the nearest boundary value.

# Author
Alex S. Gardner, JPL, Caltech.
"""
function interval_average(result, t1::AbstractVector, t2::AbstractVector)
    signal = result.signal
    t_out  = Float64.(yeardecimal.(dims(signal, :Ti).val))
    s_out  = Float64.(signal.data)

    t1_yr  = Float64.(yeardecimal.(t1))
    t2_yr  = Float64.(yeardecimal.(t2))

    # Linear interpolation at arbitrary decimal year
    function _interp(t)
        idx = searchsortedlast(t_out, t)
        idx == 0              && return s_out[1]
        idx == length(t_out)  && return s_out[end]
        t0, t1_ = t_out[idx], t_out[idx + 1]
        s0, s1_ = s_out[idx], s_out[idx + 1]
        return s0 + (s1_ - s0) * (t - t0) / (t1_ - t0)
    end

    n   = length(t1)
    avg = Vector{Float64}(undef, n)

    for i in 1:n
        a, b = t1_yr[i], t2_yr[i]
        dt   = b - a

        if dt <= 0
            avg[i] = _interp(a)
            continue
        end

        # Interior output time steps within the open interval (a, b)
        inner = findall(a .< t_out .< b)

        # Build integration nodes: interpolated boundaries + interior grid points
        t_nodes = vcat(a,           t_out[inner], b)
        s_nodes = vcat(_interp(a),  s_out[inner], _interp(b))

        # Trapezoidal integral divided by interval length → mean value
        integral = sum(
            (t_nodes[j+1] - t_nodes[j]) * (s_nodes[j] + s_nodes[j+1]) / 2
            for j in 1:(length(t_nodes) - 1)
        )
        avg[i] = integral / dt
    end

    return avg
end
