using Dates
using LinearAlgebra
using LossFunctions: L1DistLoss, L2DistLoss, HuberLoss, deriv
using LossFunctions.Traits: DistanceLoss

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
    _irls_weights(r, loss, ε)

Compute IRLS weights for robust regression.

For L1DistLoss: w[i] = 1 / (|r[i]| + ε)  (standard IRLS for L1)
For smooth losses: w[i] = 1 / (|∂L/∂r[i]| + ε)  (derivative-based)

The L1 case requires special handling because sign(r) has constant magnitude,
which would produce uniform weights. The correct IRLS weight for L1 is
inversely proportional to the residual magnitude.

# Arguments
- `r::AbstractVector`: Residual vector
- `loss::DistanceLoss`: Loss function from LossFunctions.jl
- `ε::Float64`: Regularization parameter to prevent division by zero

# Returns
Vector of IRLS weights

# Examples
For L1 loss (L1DistLoss), w = 1 / (|r| + ε), giving robustness to outliers.
For L2 loss (L2DistLoss), deriv = r, so w = 1 / (|r| + ε).
For Huber loss, deriv is piecewise: r for |r| ≤ δ, δ·sign(r) for |r| > δ.
"""
function _irls_weights(r::AbstractVector, loss::DistanceLoss, ε::Float64)
    # Special case for L1: weights must be 1/|r| for proper robustness
    if loss isa L1DistLoss
        return @. 1.0 / (abs(r) + ε)
    end

    # For smooth losses (Huber, L2, etc), use derivative-based formula
    derivs = deriv.(Ref(loss), r)
    return @. 1.0 / (abs(derivs) + ε)
end

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

# ─────────────────────────────────────────────────────────────────────────────
# Observation redundancy filtering
# ─────────────────────────────────────────────────────────────────────────────

"""
    redundancy_filter(interval_error, interval_start, interval_end;
                      interval_bins=[Day(0), Day(16), Day(32), Day(64), Day(128), Day(256)],
                      temporal_overlap=0.5,
                      bin_count_threshold=3)

Pre-filter redundant observations using interval-stratified sliding temporal windows.

First stratifies observations by interval length (interval_end - interval_start), then
applies sliding temporal windows within each interval bin. Temporal window parameters
are **automatically derived** from the interval bin upper edges: window width is 2×
the upper edge, and step size is controlled by temporal_overlap.

**Prioritizes low-uncertainty observations** - excludes high-error observations first.

# Algorithm
1. Partition observations into interval bins based on interval_end - interval_start
   (e.g., 0-16d, 16-32d, 32-64d, 64-128d, 128-256d, 256d+)
2. Within each interval bin:
   - Calculate temporal window width = 2 × interval_bin_upper_edge
   - Calculate temporal step = window_width × (1 - temporal_overlap)
   - Apply sliding temporal window based on mid_date = interval_start + (interval_end - interval_start)/2
   - Keep top `bin_count_threshold` observations per window ranked by quality (inverse error)
3. An observation can be selected by multiple overlapping windows → keep if selected by ANY window

# Arguments
- `interval_error::Vector{<:Real}`: Combined observation error (e.g., sqrt(vx_err² + vy_err²))
- `interval_start::Vector{<:Dates.TimeType}`: Start time of each observation interval (t1)
- `interval_end::Vector{<:Dates.TimeType}`: End time of each observation interval (t2)
- `interval_bins`: Interval length binning specification (default: [Day(0), Day(16), Day(32), Day(64), Day(128), Day(256)])
  - `Period`: Uniform bin width (e.g., Day(32) creates bins [0-32d, 32-64d, ...])
  - `Vector{<:Period}`: Explicit interval bin edges
- `temporal_overlap::Float64`: Fraction of overlap between successive temporal windows (default: 0.5 for 50% overlap).
  Controls temporal step = window_width × (1 - temporal_overlap). Range: [0.0, 1.0).
- `bin_count_threshold::Union{Int, Vector{Int}}`: Maximum observations per (temporal_window × interval_bin) (default: 3).
  - `Int`: Same threshold for all interval bins
  - `Vector{Int}`: Per-bin thresholds; length must equal number of interval bins

# Returns
- `keep::BitVector`: True for observations to retain, false to exclude

# Example
```julia

# 1. Default parameters (50% overlap)
keep = redundancy_filter(interval_error, t1, t2)

# 2. More aggressive overlap (75% → smaller steps, more windows)
keep = redundancy_filter(interval_error, t1, t2; temporal_overlap=0.75)

# 3. No overlap (step = window width)
keep = redundancy_filter(interval_error, t1, t2; temporal_overlap=0.0)

# 4. Uniform interval bins with custom overlap
keep = redundancy_filter(interval_error, t1, t2;
    interval_bins=Day(30),  # Creates bins [0-30, 30-60, 60-90, ...]
    temporal_overlap=0.6)   # 60% overlap

# 5. Different thresholds per interval bin
keep = redundancy_filter(interval_error, t1, t2;
    interval_bins=[Day(0), Day(16), Day(32), Day(128)],
    bin_count_threshold=[5, 3, 2])  # More for short intervals, fewer for long

# 6. Custom interval bins with no overlap
keep = redundancy_filter(interval_error, t1, t2;
    interval_bins=[Day(0), Day(12), Day(24), Day(48)],
    temporal_overlap=0.0,
    bin_count_threshold=4)
```

# Temporal Window Derivation Example
With `interval_bins = [Day(0), Day(16), Day(32), Day(64)]` and `temporal_overlap = 0.5`:

- **Bin 1: [Day(0), Day(16)]**
  - Upper edge = 16 days
  - Window width = 2 × 16 = 32 days
  - Step = 32 × (1 - 0.5) = 16 days (50% overlap)

- **Bin 2: [Day(16), Day(32)]**
  - Upper edge = 32 days
  - Window width = 2 × 32 = 64 days
  - Step = 64 × (1 - 0.5) = 32 days (50% overlap)

- **Bin 3: [Day(32), Day(64)]**
  - Upper edge = 64 days
  - Window width = 2 × 64 = 128 days
  - Step = 128 × (1 - 0.5) = 64 days (50% overlap)

# Expected Reduction
With defaults (6 interval bins, temporal_overlap=0.5, threshold=3) over 10 years:
- Temporal windows scale with interval length (short intervals → small windows, long intervals → large windows)
- Typical: ~5,000-7,000 observations (overlapping windows select same obs multiple times)
- Reduction: 10k-100k → 2k-12k (70-90% reduction)

# User Control
- **More aggressive filtering** (fewer observations):
  - Fewer interval bins: `interval_bins = [Day(0), Day(64), Day(256)]`
  - Lower threshold: `bin_count_threshold = 1` or `[1, 1, 1]`
  - Less overlap: `temporal_overlap = 0.0` (non-overlapping windows)
- **Less aggressive filtering** (more observations):
  - More interval bins: `interval_bins = [Day(0), Day(12), Day(24), Day(48), Day(96), Day(192)]`
  - Higher threshold: `bin_count_threshold = 5` or `[6, 5, 4, 3, 2, 1]`
  - More overlap: `temporal_overlap = 0.75` (75% overlap → smaller steps)

# Author
Alex S. Gardner, JPL, Caltech.
"""
function redundancy_filter(interval_error::Vector{<:Real},
                           interval_start::Vector{<:Dates.TimeType},
                           interval_end::Vector{<:Dates.TimeType};
                           interval_bins::Union{Period, AbstractVector{<:Period}}=[Day(0), Day(16), Day(32), Day(64), Day(128), Day(256)],
                           temporal_overlap::Float64=0.5,
                           bin_count_threshold::Union{Int, Vector{Int}}=3)
    n = length(interval_error)

    # Validate inputs
    length(interval_start) == n || throw(ArgumentError("interval_start length must match interval_error"))
    length(interval_end) == n || throw(ArgumentError("interval_end length must match interval_error"))
    0.0 <= temporal_overlap < 1.0 || throw(ArgumentError("temporal_overlap must be in [0.0, 1.0)"))

    # Compute mid-date and interval duration
    # Handle Date vs DateTime: compute mid-date without Period division
    interval_duration = interval_end .- interval_start

    # Compute mid_date by converting to milliseconds, dividing, then converting back
    T = eltype(interval_start)
    if T <: Date
        # For Date: convert to DateTime, compute midpoint, convert back to Date
        mid_date = Date.(DateTime.(interval_start) .+ Millisecond.(Dates.value.(interval_duration) .÷ 2))
    else
        # For DateTime: can work with Millisecond directly
        mid_date = interval_start .+ Millisecond.(Dates.value.(interval_duration) .÷ 2)
    end

    # ── Handle interval_bins: Period → vector of bin edges ──────────────────
    interval_bin_edges = if interval_bins isa Period
        # Create uniform bins from 0 to max interval duration
        # Convert to days for consistent comparison
        max_duration_days = maximum(Dates.value.(interval_duration)) / 86_400_000.0
        bin_width_days = Dates.value(convert(Millisecond, interval_bins)) / 86_400_000.0

        edges = Float64[0.0]
        current = bin_width_days
        while current <= max_duration_days
            push!(edges, current)
            current += bin_width_days
        end
        edges
    else
        # Convert Period bin edges to days (Float64)
        [Dates.value(convert(Millisecond, edge)) / 86_400_000.0 for edge in interval_bins]
    end

    # Convert interval_duration to days (Float64) for comparison
    interval_duration_days = Dates.value.(interval_duration) ./ 86_400_000.0

    # ── Parse bin_count_threshold: Int → vector of thresholds ───────────────
    n_bins = length(interval_bin_edges)
    thresholds = if bin_count_threshold isa Int
        fill(bin_count_threshold, n_bins)
    else
        bin_count_threshold
    end

    # Validate threshold vector length
    length(thresholds) == n_bins || throw(ArgumentError(
        "bin_count_threshold vector length ($(length(thresholds))) must match number of interval bins ($n_bins)"))

    # Assign each observation to an interval bin
    interval_bin_id = zeros(Int, n)
    for i in 1:n
        # Find which interval bin this observation belongs to
        for bin_idx in 1:n_bins
            if bin_idx == n_bins
                # Last bin: includes all intervals >= this threshold
                if interval_duration_days[i] >= interval_bin_edges[bin_idx]
                    interval_bin_id[i] = bin_idx
                    break
                end
            else
                # Regular bin: [bin_edge, next_bin_edge)
                if interval_bin_edges[bin_idx] <= interval_duration_days[i] < interval_bin_edges[bin_idx+1]
                    interval_bin_id[i] = bin_idx
                    break
                end
            end
        end
        # If no bin assigned (interval < first bin edge), assign to bin 1
        if interval_bin_id[i] == 0
            interval_bin_id[i] = 1
        end
    end

    # Initialize keep set
    keep_indices = Set{Int}()

    # Process each interval bin independently
    for bin_idx in 1:n_bins
        # Get observations in this interval bin
        in_bin = interval_bin_id .== bin_idx
        bin_indices = findall(in_bin)

        if isempty(bin_indices)
            continue
        end

        # Get time range for this interval bin
        bin_mid_dates = mid_date[bin_indices]

        # Get bin-specific threshold
        threshold = thresholds[bin_idx]

        # ── Calculate temporal window parameters from interval bin upper edge ─────
        # For the last bin, use its lower edge (since there's no upper edge)
        if bin_idx == n_bins
            # Last bin: use the current bin edge as the characteristic scale
            upper_edge_days = interval_bin_edges[bin_idx]
        else
            # Regular bin: use the upper edge
            upper_edge_days = interval_bin_edges[bin_idx + 1]
        end

        temporal_window_days = 2.0 * upper_edge_days
        temporal_step_days = temporal_window_days * (1.0 - temporal_overlap)

        # ── Apply sliding temporal window ─────────────────────────────────────
        start_date = minimum(bin_mid_dates)
        end_date = maximum(bin_mid_dates)

        # Convert to DateTime if needed (Date doesn't support subsecond arithmetic)
        is_date_type = start_date isa Date
        if is_date_type
            start_date = DateTime(start_date)
            end_date = DateTime(end_date)
            bin_mid_dates_dt = DateTime.(bin_mid_dates)
        else
            bin_mid_dates_dt = bin_mid_dates
        end

        window_start = start_date
        while window_start <= end_date
            # Add window width using Millisecond for precision
            window_end = window_start + Millisecond(round(Int, temporal_window_days * 86_400_000.0))

            # Find observations in this (temporal_window, interval_bin) combination
            in_window = (bin_mid_dates_dt .>= window_start) .& (bin_mid_dates_dt .<= window_end)
            window_local_indices = findall(in_window)
            window_global_indices = bin_indices[window_local_indices]

            if length(window_global_indices) <= threshold
                # Keep all in this window
                union!(keep_indices, window_global_indices)
            else
                # Rank by quality: LOWEST ERROR FIRST (inverse error = quality score)
                scores = 1.0 ./ interval_error[window_global_indices]

                # Select top threshold observations
                best_local = partialsortperm(scores, 1:threshold, rev=true)
                union!(keep_indices, window_global_indices[best_local])
            end

            # Advance window
            window_start += Millisecond(round(Int, temporal_step_days * 86_400_000.0))
        end
    end

    # Create mask
    keep = falses(n)
    keep[collect(keep_indices)] .= true

    return keep
end
