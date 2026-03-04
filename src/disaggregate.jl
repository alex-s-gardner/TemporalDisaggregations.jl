using Dates

"""
    disaggregate(aggregate_values, interval_start, interval_end;
                   method      = :spline,
                   loss_norm   = :L2,
                   output_step = Month(1),
                   kwargs...)

Reconstruct an instantaneous time series from interval-averaged observations,
dispatching to one of three underlying methods controlled by `method`.

# Arguments
- `aggregate_values`: Vector of n observed averages over each interval.
- `interval_start`: Interval start times as `Date` or `DateTime` values.
- `interval_end`: Interval end times as `Date` or `DateTime` values.
- `method`: Reconstruction algorithm (default `:spline`):
  - `:spline`   — Quartic B-spline antiderivative fit (see `disaggregate_spline`).
    Key kwargs: `smoothness`, `n_knots`, `penalty_order`, `tension`.
  - `:sinusoid` — Parametric mean + trend + seasonal + inter-annual model
    (see `disaggregate_sinusoid`).
    Key kwargs: `smoothness_interannual`.
  - `:gp`       — Sparse inducing-point Gaussian Process (see `disaggregate_gp`).
    Key kwargs: `kernel`, `obs_noise`, `n_quad`.
- `loss_norm`: Loss function, shared by all methods. `:L2` (default) = weighted
  least-squares; `:L1` = robust LAD via IRLS (suppresses blunders automatically).
- `output_step`: Temporal resolution of the output `dates` grid as a `Dates.Period`
  (e.g. `Day(1)`, `Week(1)`, `Month(3)`). Default `Month(1)`. For `:gp`, inducing
  points are always kept on a monthly grid; finer output grids use an extra kriging step.
- `kwargs...`: All remaining keyword arguments are forwarded unchanged to the
  underlying method function. Pass any method-specific kwarg here.

# Returns
`DimStack` with layers `values` and `std` (both `DimArray` with `Ti(dates)` dimension)
for all methods. Access patterns:
- `result.values`               — `DimArray` of posterior mean
- `result.std`                  — `DimArray` of posterior std (≥ 0)
- `Array(result.values)`        — plain `Vector{Float64}`
- `collect(dims(result.values, Ti))` — `Vector{Date}` output grid
- `metadata(result)`            — method-specific `Dict`; sinusoid includes
  `:mean`, `:trend`, `:amplitude`, `:phase`, `:interannual`

# Examples
```julia
# Switch methods by changing one kwarg:
r_spline = disaggregate(y, t1, t2; method = :spline, smoothness = 1e-3)
r_sin    = disaggregate(y, t1, t2; method = :sinusoid)
r_gp     = disaggregate(y, t1, t2; method = :gp, obs_noise = 4.0)

# L1 loss for robustness to blunders (works with any method):
r_robust = disaggregate(y, t1, t2; method = :gp, loss_norm = :L1, obs_noise = 4.0)
```
"""
function disaggregate(aggregate_values::AbstractVector,
                        interval_start::AbstractVector{<:Dates.TimeType},
                        interval_end::AbstractVector{<:Dates.TimeType};
                        method::Symbol        = :spline,
                        loss_norm::Symbol     = :L2,
                        output_step::Dates.Period = Month(1),
                        kwargs...)
    method ∈ (:spline, :sinusoid, :gp) ||
        throw(ArgumentError(
            "method must be :spline, :sinusoid, or :gp; got :$method."))
    loss_norm ∈ (:L1, :L2) ||
        throw(ArgumentError("loss_norm must be :L1 or :L2; got :$loss_norm."))

    if method == :spline
        return disaggregate_spline(aggregate_values, interval_start, interval_end;
                                     loss_norm, output_step, kwargs...)
    elseif method == :sinusoid
        return disaggregate_sinusoid(aggregate_values, interval_start, interval_end;
                                       loss_norm, output_step, kwargs...)
    else  # :gp
        return disaggregate_gp(aggregate_values, interval_start, interval_end;
                                 loss_norm, output_step, kwargs...)
    end
end
