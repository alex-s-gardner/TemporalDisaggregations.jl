using Dates
using LinearAlgebra

# ─────────────────────────────────────────────────────────────────────────────
# Time helpers
# ─────────────────────────────────────────────────────────────────────────────

yeardecimal(d::Dates.Date)::Float64 =
    year(d) + (d - Date(year(d), 1, 1)).value / (isleapyear(year(d)) ? 366.0 : 365.0)

yeardecimal(dt::Dates.DateTime)::Float64 =
    year(dt) + (dt - DateTime(year(dt), 1, 1)).value /
               (86_400_000.0 * (isleapyear(year(dt)) ? 366 : 365))

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
