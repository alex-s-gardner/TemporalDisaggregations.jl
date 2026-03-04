# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Start Julia REPL with the package loaded
julia --project=.

# Run tests
julia --project=test test/runtests.jl
# or from the Julia REPL:
# ] activate test; test TemporalDisaggregation

# Run a specific test block (from the REPL with package loaded)
# include("test/runtests.jl")

# Run the main end-to-end tutorial (generates 7 figures, requires CairoMakie, DimensionalData)
julia --project=examples examples/tutorial.jl
```

**Requirements:** Julia ≥ 1.10. Test environment (`test/Project.toml`) uses only stdlib `Test`. `CairoMakie` and `DimensionalData` are in the **main** `Project.toml` but only used in `examples/`.

## Architecture

This is a Julia package that reconstructs instantaneous time series from interval-averaged observations (temporal disaggregation). The core problem: given measurements averaged over overlapping time intervals, recover the underlying instantaneous signal.

**Exports:** `disaggregate` and `decimal_year`.

**Single public API:** `disaggregate(aggregate_values, interval_start, interval_end; method, loss_norm, kwargs...)` in `src/disaggregate.jl` — validates inputs and dispatches to one of three method implementations.

**`decimal_year(d)`** in `src/utils.jl` — converts a `Date` or `DateTime` to a decimal year (leap-year-aware). E.g. `decimal_year(Date(2020, 7, 2)) ≈ 2020.5`.

**Time representation:** All interval boundaries are decimal years (e.g. `2020.5` = mid-2020).

**Output / return type:** `disaggregate()` returns a `DimStack` for all three methods:
```julia
DimStack(
    (signal = DimArray(values, Ti(dates)),
     std    = DimArray(std_vec, Ti(dates)));
    metadata = Dict(...)
)
```
- `output_period` kwarg (default `Month(1)`) controls grid spacing — any `Dates.Period` is valid (`Day`, `Week`, `Month`, `Year`).
- `output_start` anchors the grid start.
- When `output_period ≠ Month(1)` for the GP method, inducing points stay monthly and a kriging step predicts at the arbitrary grid (extra O(m²) solve).

**Three methods** (each in its own file):

- `src/disaggregate_spline.jl` — Quartic B-spline antiderivative fit. Models F(t) as a B-spline so that F(t2ᵢ) − F(t1ᵢ) = observed area; instantaneous signal = F′(t). Uses P-spline regularization with optional tension penalty. Supports `outlier_rejection`.
  - Metadata keys: `:method=>:spline`, `:smoothness`, `:n_knots`, `:penalty_order`, `:tension`, `:loss_norm`, `:output_period`

- `src/disaggregate_sinusoid.jl` — Parametric model: mean + trend + per-year anomalies + annual sinusoid. Design matrix is constructed analytically (exact interval integrals, no quadrature); this is the fastest method. Supports `outlier_rejection`.
  - Metadata keys: `:method=>:sinusoid`, `:mean`, `:trend`, `:amplitude`, `:phase`, `:interannual` (`Dict{Int,Float64}`), `:smoothness_interannual`, `:loss_norm`, `:output_period`

- `src/disaggregate_gp.jl` — Sparse inducing-point GP (DTC approximation) on a monthly grid. Uses Gauss-Legendre quadrature (`FastGaussQuadrature`) to build the integral cross-kernel matrix via `Threads.@threads`. Matrix inversion lemma avoids forming the n×n observation covariance. Does **not** support `outlier_rejection`.
  - Metadata keys: `:method=>:gp`, `:kernel`, `:obs_noise`, `:n_quad`, `:loss_norm`, `:output_period`

**Shared L1/L2 loss:** All three methods implement optional L1 loss via Iteratively Reweighted Least Squares (IRLS, max 50 iterations, tolerance 1e-8 infinity-norm on relative weight change). The `loss_norm` kwarg is accepted by both `disaggregate()` and each underlying function.

## Helper functions (`src/utils.jl`)

- `decimal_year(d)` — **exported**; `Date`/`DateTime` → decimal year, leap-year-aware.
- `_date_grid(t_min, t_max, step; output_start)` — general grid generator with leap-year handling.
- `_monthly_decimal_year_grid(t_min, t_max)` — specialised monthly wrapper used by all three methods.
- `_difference_matrix(m, r)` — builds the r-th order difference matrix for P-spline regularization.
- `_irls_weights` / `_irls_converged` — shared IRLS helpers used by all methods.

**Other helpers** (in their respective method files):
- `_interval_sin_integral`, `_interval_cos_integral`, `_year_overlap_fraction` in `disaggregate_sinusoid.jl` — closed-form interval averages for the sinusoid design matrix.

## Internal conventions

- Intervals are sorted chronologically inside each method.
- Smoothness λ is scaled by ‖C'C‖/n to be dimensionless across datasets.

## Testing patterns

Tests in `test/runtests.jl` use helper generators `make_monthly_intervals()` and `exact_average()` to create synthetic data with known ground truth. Coverage: input validation, signal recovery, L1/L2 agreement, output grid variations, method-specific kwargs, edge cases (tension, blunders, kriging paths).

## Examples

`examples/tutorial.jl` is the main end-to-end demo. It generates 7 figures covering all three methods, parameter sweeps, kernel designs, L1 vs L2 robustness, and output grid options.
