"""
    disaggregate(method, aggregate_values, interval_start, interval_end;
                 loss_norm=L2DistLoss(), output_period=Month(1), output_start=nothing,
                 output_end=nothing, weights=nothing, irls_tol=1e-8, irls_max_iter=50)

Reconstruct an instantaneous time series from interval-averaged observations.

# Arguments
- `method::DisaggregationMethod`: Algorithm configuration. One of:
  - `Spline(; smoothness=1e-1, penalty_order, tension)`
  - `Sinusoid(; smoothness_interannual)`
  - `GP(; kernel, obs_noise, n_quad)`
- `aggregate_values`: Vector of n observed averages over each interval.
- `interval_start`, `interval_end`: Interval boundaries as `Date`/`DateTime`.
- `loss_norm::DistanceLoss = L2DistLoss()`: Loss function from LossFunctions.jl.
  Common choices: `L2DistLoss()` (least squares), `L1DistLoss()` (robust to outliers),
  `HuberLoss(δ)` (hybrid - L2 for small residuals, L1 for large). For Huber loss,
  specify the threshold δ (default 1.345 achieves 95% efficiency at the normal distribution).
  **Note**: Robust losses (L1, Huber) use IRLS with a fixed penalty term. The same `smoothness`
  parameter may produce different effective smoothness for different loss functions. Tune
  `smoothness` separately for each loss type if matching visual smoothness is important.
- `output_period::Dates.Period = Month(1)`: Output grid spacing.
- `output_start`: Grid anchor `Date` or `DateTime` (default `nothing`).
- `output_end`: Last date of the output grid as `Date` or `DateTime`. Defaults
  to the end of the data domain.
- `weights`: Optional vector of n positive per-observation weights (e.g. `1 ./ σ²`).
  `nothing` (default) uses uniform weights. For robust losses (L1, Huber), these are
  multiplied element-wise with the IRLS weights at each iteration.
- `irls_tol::Float64 = 1e-8`: Convergence tolerance for IRLS iterations (only used for
  non-L2 losses). Smaller values (e.g., 1e-10) give more accurate solutions but take
  longer; larger values (e.g., 1e-6) converge faster but may be less accurate. The
  tolerance controls the relative parameter change: iterations stop when
  `max(|x_new - x|) / (‖x‖ + 1e-10) < irls_tol`.
- `irls_max_iter::Int = 50`: Maximum number of IRLS iterations (only used for non-L2
  losses). Increase if IRLS does not converge within 50 iterations for difficult problems;
  decrease for faster (but potentially less accurate) solutions.

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

# Fast L1 solution with looser tolerance (2-3× faster)
result = disaggregate(Spline(), y, t1, t2; loss_norm=L1DistLoss(), irls_tol=1e-6)

# High-precision L1 solution with tighter tolerance
result = disaggregate(Spline(), y, t1, t2; loss_norm=L1DistLoss(), irls_tol=1e-10)

# Fast L1 with fewer IRLS iterations (for very large problems)
result = disaggregate(Spline(), y, t1, t2; loss_norm=L1DistLoss(), irls_max_iter=20)

# More iterations for difficult convergence cases
result = disaggregate(Spline(), y, t1, t2; loss_norm=HuberLoss(1.0), irls_max_iter=100)
```
"""
function disaggregate end
