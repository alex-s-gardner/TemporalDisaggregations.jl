using Dates
using LinearAlgebra

"""
    _monthly_yeardecimal_grid(t_min, t_max)

Return `(dates, yeardecimals)` for a monthly `Date` grid covering `[t_min, t_max]`,
where both arguments are decimal years (e.g. `2020.5` = mid-2020).
"""
function _monthly_yeardecimal_grid(t_min::Real, t_max::Real)
    y0 = floor(Int, t_min)
    m0 = clamp(floor(Int, (t_min - y0) * 12) + 1, 1, 12)
    y1 = floor(Int, t_max)
    m1 = clamp(floor(Int, (t_max - y1) * 12) + 1, 1, 12)
    dates = collect(Date(y0, m0, 1):Month(1):Date(y1, m1, 1))
    times = [year(d) + (month(d) - 1) / 12.0 for d in dates]
    return dates, times
end

"""
    _date_grid(t_min, t_max, step; output_start=nothing)

Return `(dates, yeardecimals)` for a `Date` grid covering `[t_min, t_max]` with the
given `step` (any `Dates.Period`), where both bounds are decimal years.

If `output_start` is provided, it anchors the grid:
- For `Month` steps: the day-of-month from `output_start` is used (e.g. 15th of each month).
- For other steps: `output_start` is used as the literal start date.
"""
function _date_grid(t_min::Real, t_max::Real, step::Dates.Period;
                    output_start::Union{Dates.TimeType,Nothing} = nothing)
    sub_day = step isa Union{Hour, Minute, Second, Millisecond}

    yr0 = floor(Int, t_min); ndays0 = isleapyear(yr0) ? 366 : 365
    d0_date = Date(yr0, 1, 1) + Day(floor(Int, (t_min - yr0) * ndays0))
    if isnothing(output_start)
        d_start = sub_day ? DateTime(d0_date) : d0_date
    elseif step isa Month
        m0      = clamp(floor(Int, (t_min - yr0) * 12) + 1, 1, 12)
        d_start = Date(yr0, m0, min(day(output_start), daysinmonth(yr0, m0)))
    else
        d_start = sub_day && output_start isa Date ? DateTime(output_start) : output_start
    end
    yr1 = floor(Int, t_max); ndays1 = isleapyear(yr1) ? 366 : 365
    d1_date = Date(yr1, 1, 1) + Day(floor(Int, (t_max - yr1) * ndays1))
    d_end   = sub_day ? DateTime(d1_date) : d1_date
    dates = collect(d_start:step:d_end)
    times = yeardecimal.(dates)
    return dates, times
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
