# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Start Julia REPL with the package loaded
julia --project=.

# Run tests
julia --project=test test/runtests.jl
# or from the Julia REPL:
# ] activate test; test TemporalDisaggregations

# Run a specific test block (from the REPL with package loaded)
# include("test/runtests.jl")

# Run the main end-to-end tutorial (generates figures, requires CairoMakie, DimensionalData)
julia --project=examples examples/tutorial.jl

# Run benchmarks (Spline / Sinusoid / GP at 1e4, 1e5, 1e6 observations)
julia --project=. --threads=auto benchmarks/benchmark.jl

# Build docs (requires DocumenterVitepress)
julia --project=docs docs/make.jl
```

**Requirements:** Julia ≥ 1.10. Test environment (`test/Project.toml`) adds only stdlib `Test`; the test file also imports `TemporalDisaggregations`, `DimensionalData`, `LinearAlgebra`, and `Statistics` from the main `Project.toml`. `CairoMakie` is in the main `Project.toml` but only used in `examples/`.

**Performance optimization:** Optional BLAS acceleration via package extensions. Add `using AppleAccelerate` (macOS) or `using MKL` to redirect OpenBLAS → native accelerated libraries. No code changes needed; the extension auto-activates when the corresponding package is loaded.

## File Structure

```
src/
├── TemporalDisaggregations.jl    # main module, exports, includes
├── methods.jl                     # abstract type and method structs (Spline, Sinusoid, GP)
├── disaggregate.jl                # generic function declaration + docstring
├── disaggregate_spline.jl         # Spline method implementation
├── disaggregate_sinusoid.jl       # Sinusoid method implementation  
├── disaggregate_gp.jl             # GP method implementation
├── utils.jl                       # shared helpers (IRLS, date grids, P-spline matrices)
└── precompile.jl                  # precompilation directives

ext/                               # package extensions for BLAS optimization
├── TemporalDisaggregationsAppleAccelerateExt.jl
└── TemporalDisaggregationsMKLExt.jl

test/runtests.jl                   # all tests in one file
examples/tutorial.jl               # main demo (generates 8 figures)
benchmarks/benchmark.jl            # performance benchmarks
docs/                              # DocumenterVitepress docs
```

## Architecture

This is a Julia package that reconstructs instantaneous time series from interval-averaged observations (temporal disaggregation). The core problem: given measurements averaged over overlapping time intervals, recover the underlying instantaneous signal.

**Exports:** `disaggregate`, `yeardecimal`, `interval_average`, `redundancy_filter`, `DisaggregationMethod`, `Spline`, `Sinusoid`, `GP`.

**Public API:** `disaggregate(method::DisaggregationMethod, aggregate_values, interval_start, interval_end; loss_norm=L2DistLoss(), output_period=Month(1), output_start=nothing, output_end=nothing, weights=nothing)` — dispatches on the algorithm struct type. Method-specific parameters live in the struct; shared kwargs (`loss_norm`, `output_period`, `output_start`, `output_end`, `weights`) stay as function kwargs. `weights` is a length-n vector of positive per-observation weights (e.g. `1 ./ σ²_obs`); for robust losses they are multiplied element-wise with the IRLS weights.

**Algorithm structs** (`src/methods.jl`):
```julia
abstract type DisaggregationMethod end
Spline    <: DisaggregationMethod   # smoothness, n_knots, penalty_order, tension
Sinusoid  <: DisaggregationMethod   # smoothness_interannual
GP        <: DisaggregationMethod   # kernel, obs_noise, n_quad
```
All use `@kwdef` with sensible defaults. `GP.kernel` is untyped (`Any`) because KernelFunctions.jl compositions produce deeply nested parametric types.

**`yeardecimal(d)`** — re-exported from `DateFormats.jl`; converts a `Date` or `DateTime` to a decimal year (leap-year-aware). E.g. `yeardecimal(Date(2020, 7, 2)) ≈ 2020.5`.

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
- `output_start` anchors the grid start; `output_end` caps the grid end.
- For the GP method, the inducing grid is always at `_half_period(output_period)` resolution, and the output is always kriged from the inducing grid (extra O(m²) solve).

**`src/disaggregate.jl`** — declares the `disaggregate` generic function with its full docstring. Method implementations live in the per-method files below.

**Three method implementations** (each dispatches on its struct):

- `src/disaggregate_spline.jl` — `disaggregate(m::Spline, ...)`. Quartic B-spline antiderivative fit. Models F(t) as a B-spline so that F(t2ᵢ) − F(t1ᵢ) = observed area; instantaneous signal = F′(t). Uses P-spline regularization with optional tension penalty.

- `src/disaggregate_sinusoid.jl` — `disaggregate(m::Sinusoid, ...)`. Parametric model: mean + trend + per-year anomalies + annual sinusoid. Design matrix is constructed analytically (exact interval integrals, no quadrature); fastest method.

- `src/disaggregate_gp.jl` — `disaggregate(m::GP, ...)`. Sparse inducing-point GP (DTC approximation); inducing grid is 2× finer than `output_period` (floored at `Day(1)`). Uses Gauss-Legendre quadrature (`FastGaussQuadrature`) to build the integral cross-kernel matrix via `Threads.@threads` (use `--threads=auto` for best performance). Matrix inversion lemma avoids forming the n×n observation covariance. Always krigs from the inducing grid to the output grid.

**Robust loss functions:** All three methods support any `DistanceLoss` from [LossFunctions.jl](https://github.com/JuliaML/LossFunctions.jl) via the `loss_norm` kwarg. For losses other than `L2DistLoss()`, IRLS is used (max 50 iterations, tolerance 1e-8). IRLS weights: `w = 1 / (|∂L/∂r| + ε)` where `∂L/∂r = deriv(loss, r)`. Common choices: `L1DistLoss()` (robust to outliers), `HuberLoss(δ)` (hybrid), `L2DistLoss()` (standard LS).

**`std` semantics are identical across all methods**: spatially-varying sandwich std
`std(t*) = σ̂ · sqrt(q(t*))` where `σ̂` is the weighted residual RMS and `q(t*)` is a
dimensionless coverage factor (lower in dense regions, higher in sparse regions).
When using robust losses (non-L2), computed from the final IRLS solution.

## Helper functions (`src/utils.jl`)

- `yeardecimal(d)` — **exported**; re-exported from `DateFormats`; `Date`/`DateTime` → decimal year, leap-year-aware.
- `_date_grid(t_min, t_max, step; output_start)` — general grid generator with leap-year handling.
- `_half_period(p)` — returns a period ≈ half the size of `p`, floored at `Day(1)`; used by the GP method to set inducing grid spacing at 2× the output resolution.
- `_difference_matrix(m, r)` — builds the r-th order difference matrix for P-spline regularization.
- `_irls_weights` / `_irls_converged` — shared IRLS helpers used by all methods.

**Post-processing:**
- `interval_average(result, t1, t2)` — **exported**; trapezoidal re-integration of a `disaggregate` result over arbitrary intervals. Useful for computing residuals or forward-checking fit quality.
- `redundancy_filter(interval_error, interval_start, interval_end; interval_bins, temporal_overlap, bin_count_threshold)` — **exported**; pre-filter redundant observations using interval-stratified sliding temporal windows. Prioritizes low-uncertainty observations. Returns a `BitVector` mask of observations to keep.

**Other helpers** (in their respective method files):
- `_interval_sin_integral`, `_interval_cos_integral`, `_year_overlap_fraction` in `disaggregate_sinusoid.jl` — closed-form interval averages for the sinusoid design matrix.

## Internal conventions

- Intervals are sorted chronologically inside each method.
- Smoothness λ is scaled by ‖C'C‖/n to be dimensionless across datasets.

## Testing patterns

Tests in `test/runtests.jl` use `const TD = TemporalDisaggregations` to access internal (unexported) functions. Helper generators `make_monthly_intervals()` and `exact_average()` create synthetic data with known ground truth. Coverage: input validation, signal recovery, L1/L2 agreement, output grid variations, method-specific kwargs, edge cases (tension, blunders, kriging paths).

## Examples

`examples/tutorial.jl` is the main end-to-end demo. It generates 8 figures covering all three methods, parameter sweeps, kernel designs, L1 vs L2 robustness, output grid options, and per-observation weights (heteroscedastic noise).
