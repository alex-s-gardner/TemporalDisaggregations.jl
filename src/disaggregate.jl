"""
    disaggregate(method, aggregate_values, interval_start, interval_end;
                 loss_norm=L2DistLoss(), output_period=Month(1), output_start=nothing,
                 output_end=nothing, weights=nothing)

Reconstruct an instantaneous time series from interval-averaged observations.

# Arguments
- `method::DisaggregationMethod`: Algorithm configuration. One of:
  - `Spline(; smoothness=1e-1, n_knots, penalty_order, tension)`
  - `Sinusoid(; smoothness_interannual)`
  - `GP(; kernel, obs_noise, n_quad)`
- `aggregate_values`: Vector of n observed averages over each interval.
- `interval_start`, `interval_end`: Interval boundaries as `Date`/`DateTime`.
- `loss_norm::DistanceLoss = L2DistLoss()`: Loss function from LossFunctions.jl.
  Common choices: `L2DistLoss()` (least squares), `L1DistLoss()` (robust to outliers),
  `HuberLoss(δ)` (hybrid - L2 for small residuals, L1 for large). For Huber loss,
  specify the threshold δ (default 1.345 achieves 95% efficiency at the normal distribution).
- `output_period::Dates.Period = Month(1)`: Output grid spacing.
- `output_start`: Grid anchor `Date` or `DateTime` (default `nothing`).
- `output_end`: Last date of the output grid as `Date` or `DateTime`. Defaults
  to the end of the data domain.
- `weights`: Optional vector of n positive per-observation weights (e.g. `1 ./ σ²`).
  `nothing` (default) uses uniform weights. For robust losses (L1, Huber), these are
  multiplied element-wise with the IRLS weights at each iteration.

# Returns
`DimStack` with `:signal` and `:std` layers indexed by `Ti(dates)`.
For all methods, `:std` is the spatially-varying sandwich standard deviation:
`std(t*) = σ̂ · sqrt(q(t*))`, where `σ̂` is the weighted residual RMS of predicted
vs. observed interval averages and `q(t*)` is a dimensionless coverage factor that is
smaller where observations are dense and larger where they are sparse.

# Examples
```julia
using LossFunctions

result = disaggregate(Spline(smoothness=1e-3), y, t1, t2)
result = disaggregate(GP(obs_noise=4.0), y, t1, t2; loss_norm=L1DistLoss())
result = disaggregate(Sinusoid(), y, t1, t2; output_period=Day(1))
result = disaggregate(Spline(), y, t1, t2; output_end=Date(2020, 6, 1))
result = disaggregate(Spline(), y, t1, t2; weights = 1 ./ σ²_obs)
result = disaggregate(Spline(), y, t1, t2; loss_norm=HuberLoss(1.345))
result = disaggregate(Spline(), y, t1, t2; loss_norm=HuberLoss(2.0))
```
"""
function disaggregate end
