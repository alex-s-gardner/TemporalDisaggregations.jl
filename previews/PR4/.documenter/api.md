
# API Reference {#API-Reference}

## Public API {#Public-API}
<details class='jldocstring custom-block' open>
<summary><a id='TemporalDisaggregations.disaggregate' href='#TemporalDisaggregations.disaggregate'><span class="jlbinding">TemporalDisaggregations.disaggregate</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
disaggregate(method, aggregate_values, interval_start, interval_end;
             loss_norm=:L2, output_period=Month(1), output_start=nothing)
```


Reconstruct an instantaneous time series from interval-averaged observations.

**Arguments**
- `method::DisaggregationMethod`: Algorithm configuration. One of:
  - `Spline(; smoothness, n_knots, penalty_order, tension)`
    
  - `Sinusoid(; smoothness_interannual)`
    
  - `GP(; kernel, obs_noise, n_quad)`
    
  
- `aggregate_values`: Vector of n observed averages over each interval.
  
- `interval_start`, `interval_end`: Interval boundaries as `Date`/`DateTime`.
  
- `loss_norm::Symbol = :L2`: `:L2` or `:L1` (robust to outliers via IRLS).
  
- `output_period::Dates.Period = Month(1)`: Output grid spacing.
  
- `output_start`: Grid anchor date (default `nothing`).
  

**Returns**

`DimStack` with `:signal` and `:std` layers indexed by `Ti(dates)`.

**Examples**

```julia
result = disaggregate(Spline(smoothness=1e-3), y, t1, t2)
result = disaggregate(GP(obs_noise=4.0), y, t1, t2; loss_norm=:L1)
result = disaggregate(Sinusoid(), y, t1, t2; output_period=Day(1))
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/alex-s-gardner/TemporalDisaggregations.jl" target="_blank" rel="noreferrer">source</a></Badge>

</details>


::: warning Missing docstring.

Missing docstring for `decimal_year`. Check Documenter&#39;s build log for details.

:::
