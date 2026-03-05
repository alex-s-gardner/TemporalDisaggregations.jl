
# API Reference {#API-Reference}

## Public API {#Public-API}
<details class='jldocstring custom-block' open>
<summary><a id='TemporalDisaggregations.disaggregate' href='#TemporalDisaggregations.disaggregate'><span class="jlbinding">TemporalDisaggregations.disaggregate</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
disaggregate(aggregate_values, interval_start, interval_end;
               method      = :spline,
               loss_norm   = :L2,
               output_period = Month(1),
               kwargs...)
```


Reconstruct an instantaneous time series from interval-averaged observations, dispatching to one of three underlying methods controlled by `method`.

**Arguments**
- `aggregate_values`: Vector of n observed averages over each interval.
  
- `interval_start`: Interval start times as `Date` or `DateTime` values.
  
- `interval_end`: Interval end times as `Date` or `DateTime` values.
  
- `method`: Reconstruction algorithm (default `:spline`):
  - `:spline`   — Quartic B-spline antiderivative fit (see `disaggregate_spline`). Key kwargs: `smoothness`, `n_knots`, `penalty_order`, `tension`.
    
  - `:sinusoid` — Parametric mean + trend + seasonal + inter-annual model (see `disaggregate_sinusoid`). Key kwargs: `smoothness_interannual`.
    
  - `:gp`       — Sparse inducing-point Gaussian Process (see `disaggregate_gp`). Key kwargs: `kernel`, `obs_noise`, `n_quad`.
    
  
- `loss_norm`: Loss function, shared by all methods. `:L2` (default) = weighted least-squares; `:L1` = robust LAD via IRLS (suppresses blunders automatically).
  
- `output_period`: Temporal resolution of the output `dates` grid as a `Dates.Period` (e.g. `Day(1)`, `Week(1)`, `Month(3)`). Default `Month(1)`. For `:gp`, inducing points are always kept on a monthly grid; finer output grids use an extra kriging step.
  
- `output_start`: Anchor date for the output grid. For `Month` steps, the day-of-month is used (e.g. `Date(2020,1,15)` → 15th of each month). For other steps, used as the literal start date. Default `nothing` (1st of each month for monthly output).
  
- `kwargs...`: All remaining keyword arguments are forwarded unchanged to the underlying method function. Pass any method-specific kwarg here.
  

**Returns**

`DimStack` with layers `values` and `std` (both `DimArray` with `Ti(dates)` dimension) for all methods. Access patterns:
- `result.signal`               — `DimArray` of posterior mean
  
- `result.std`                  — `DimArray` of posterior std (≥ 0)
  
- `Array(result.signal)`        — plain `Vector{Float64}`
  
- `collect(dims(result.signal, Ti))` — `Vector{Date}` output grid
  
- `metadata(result)`            — method-specific `Dict`; sinusoid includes `:mean`, `:trend`, `:amplitude`, `:phase`, `:interannual`
  

**Examples**

```julia
# Switch methods by changing one kwarg:
r_spline = disaggregate(y, t1, t2; method = :spline, smoothness = 1e-3)
r_sin    = disaggregate(y, t1, t2; method = :sinusoid)
r_gp     = disaggregate(y, t1, t2; method = :gp, obs_noise = 4.0)

# L1 loss for robustness to blunders (works with any method):
r_robust = disaggregate(y, t1, t2; method = :gp, loss_norm = :L1, obs_noise = 4.0)
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/alex-s-gardner/TemporalDisaggregations.jl" target="_blank" rel="noreferrer">source</a></Badge>

</details>


::: warning Missing docstring.

Missing docstring for `decimal_year`. Check Documenter&#39;s build log for details.

:::
