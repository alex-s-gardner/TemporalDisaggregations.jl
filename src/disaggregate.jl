"""
    disaggregate(method, aggregate_values, interval_start, interval_end;
                 loss_norm=:L2, output_period=Month(1), output_start=nothing,
                 output_end=nothing, weights=nothing)

Reconstruct an instantaneous time series from interval-averaged observations.

# Arguments
- `method::DisaggregationMethod`: Algorithm configuration. One of:
  - `Spline(; smoothness, n_knots, penalty_order, tension)`
  - `Sinusoid(; smoothness_interannual)`
  - `GP(; kernel, obs_noise, n_quad)`
- `aggregate_values`: Vector of n observed averages over each interval.
- `interval_start`, `interval_end`: Interval boundaries as `Date`/`DateTime`.
- `loss_norm::Symbol = :L2`: `:L2` or `:L1` (robust to outliers via IRLS).
- `output_period::Dates.Period = Month(1)`: Output grid spacing.
- `output_start`: Grid anchor `Date` or `DateTime` (default `nothing`).
- `output_end`: Last date of the output grid as `Date` or `DateTime`. Defaults
  to the end of the data domain.
- `weights`: Optional vector of n positive per-observation weights (e.g. `1 ./ σ²`).
  `nothing` (default) uses uniform weights. For `:L1` loss, these are multiplied
  element-wise with the IRLS weights at each iteration.

# Returns
`DimStack` with `:signal` and `:std` layers indexed by `Ti(dates)`.
For `GP`, `:std` is the Bayesian posterior standard deviation. For `Spline` and `Sinusoid`,
`:std` is a constant across the output grid equal to the residual standard deviation of
predicted vs. observed interval averages (`std(y .- ŷ)`).

# Examples
```julia
result = disaggregate(Spline(smoothness=1e-3), y, t1, t2)
result = disaggregate(GP(obs_noise=4.0), y, t1, t2; loss_norm=:L1)
result = disaggregate(Sinusoid(), y, t1, t2; output_period=Day(1))
result = disaggregate(Spline(), y, t1, t2; output_end=Date(2020, 6, 1))
result = disaggregate(Spline(), y, t1, t2; weights = 1 ./ σ²_obs)
```
"""
function disaggregate end
