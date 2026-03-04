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

# Run the example script (requires CairoMakie, DimensionalData)
julia --project=examples examples/disaggregate_example.jl
```

## Architecture

This is a Julia package that reconstructs instantaneous time series from interval-averaged observations (temporal disaggregation). The core problem: given measurements averaged over overlapping time intervals, recover the underlying instantaneous signal.

**Single public API:** `disaggregate(aggregate_values, interval_start, interval_end; method, loss_norm, kwargs...)` in `src/disaggregate.jl` — validates inputs and dispatches to one of three method implementations. Only `disaggregate` is exported; the three underlying functions are accessible but unexported.

**Time representation:** All interval boundaries are decimal years (e.g. `2020.5` = mid-2020). Output is always on a monthly `Date` grid.

**Three methods** (each in its own file):

- `src/disaggregate_spline.jl` — Quartic B-spline antiderivative fit. Models F(t) as a B-spline so that F(t2ᵢ) − F(t1ᵢ) = observed area; instantaneous signal = F′(t). Uses P-spline regularization with optional tension penalty. Supports `outlier_rejection`. Returns `(dates, values)`.

- `src/disaggregate_sinusoid.jl` — Parametric model: mean + trend + per-year anomalies + annual sinusoid. Design matrix is constructed analytically (exact interval integrals, no quadrature); this is the fastest method. Supports `outlier_rejection`. Returns `(dates, values, mean, trend, amplitude, phase, interannual)`.

- `src/disaggregate_gp.jl` — Sparse inducing-point GP (DTC approximation) on a monthly grid. Uses Gauss-Legendre quadrature (`FastGaussQuadrature`) to build the integral cross-kernel matrix via `Threads.@threads`. Matrix inversion lemma avoids forming the n×n observation covariance. Does **not** support `outlier_rejection`. Returns `(dates, values, std)`.

**Shared L1/L2 loss:** All three methods implement optional L1 loss via Iteratively Reweighted Least Squares (IRLS, up to 50 iterations, tolerance 1e-8). The `loss_norm` kwarg is accepted by both `disaggregate()` and each underlying function.

**Key shared helper:** `_monthly_decimal_year_grid(t_min, t_max)` (defined in `disaggregate_spline.jl`, used by all three methods) generates the output date grid.

**Other helpers:**
- `_difference_matrix(m, r)` in `disaggregate_spline.jl` — builds the r-th order difference matrix for P-spline regularization.
- `_interval_sin_integral`, `_interval_cos_integral`, `_year_overlap_fraction` in `disaggregate_sinusoid.jl` — closed-form interval averages for the sinusoid design matrix.

**Dependencies:** `CairoMakie` and `DimensionalData` are listed in the main `Project.toml` (not a separate dev environment) but are only used in `examples/`.
