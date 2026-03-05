# tutorial.jl — TemporalDisaggregations.jl end-to-end tutorial
#
# Run from the repository root:
#   julia --project=examples examples/tutorial.jl
#
# Produces 7 PNG files in examples/figures/.
# Uses a fixed random seed so all figures are reproducible.

using TemporalDisaggregations
using KernelFunctions
using CairoMakie
using DimensionalData: dims, Ti
import DimensionalData
using Dates
using Statistics
using Random

Random.seed!(42)

fig_dir = joinpath(@__DIR__, "figures")
mkpath(fig_dir)

# ─────────────────────────────────────────────────────────────────────────────
# Section 1 — Shared synthetic signal (2020–2024)
# ─────────────────────────────────────────────────────────────────────────────
# Ground truth: linear trend + annual sinusoid with interannual amplitude
# modulation + Gaussian noise.  All three methods are tested against this
# single signal so results are directly comparable.

t_daily    = collect(Date(2020, 1, 1):Day(1):Date(2024, 1, 1))
t_decyear  = decimal_year.(t_daily)

noise_std       = 2.0    # observation noise std-dev
trend_rate      = 3.0    # signal rise per year
seasonal_amp    = 10.0   # base seasonal amplitude
interannual_amp = 10.0    # interannual amplitude modulation
interannual_T   = 3.0    # period of interannual cycle (years)

# Amplitude-modulated annual sinusoid + linear trend + noise
amp_t  = seasonal_amp .+ interannual_amp .* sin.(2π .* (t_decyear .- 2020.0) ./ interannual_T)
signal = amp_t .* sin.(2π .* t_decyear) .+
         trend_rate .* (t_decyear .- 2022.0) .+
         noise_std .* randn(length(t_decyear))

# Compute interval-average of `signal` over [t1, t2] (both Dates).
function interval_avg(t1::Date, t2::Date)
    mask = (t1 .<= t_daily) .& (t_daily .<= t2)
    return mean(signal[mask])
end

# ── Small dataset: n = 20, intervals ~ 3–8 months long ───────────────────────
n_small = 20
ctr_s   = t_daily[rand(1:length(t_daily), n_small)]
half_s  = rand(45:120, n_small)                      # half-length in days
t1_small = max.(t_daily[1],   ctr_s .- Day.(half_s))
t2_small = min.(t_daily[end], ctr_s .+ Day.(half_s))
y_small  = [interval_avg(t1_small[i], t2_small[i]) for i in 1:n_small]

# ── Large dataset: n = 2000, intervals ~ 5–30 days long ──────────────────────
n_large = 2000
ctr_l   = t_daily[rand(1:length(t_daily), n_large)]
half_l  = rand(2:15, n_large)                        # half-length in days
t1_large = max.(t_daily[1],   ctr_l .- Day.(half_l))
t2_large = min.(t_daily[end], ctr_l .+ Day.(half_l))
y_large  = [interval_avg(t1_large[i], t2_large[i]) for i in 1:n_large]

println("Datasets ready: small n=$n_small, large n=$n_large")

# Helper: extract decimal-year time axis from a result DimStack.
# Use .val to get the raw Date array (avoids returning a DimVector which Makie
# cannot handle for band!/lines!/scatter!).
t_axis(r) = decimal_year.(dims(r, Ti).val)

# ─────────────────────────────────────────────────────────────────────────────
# Section 2 — Figure 1: Small n, all three methods
# ─────────────────────────────────────────────────────────────────────────────
# When to use:
#   Sparse, irregular, long-interval data (satellite revisits, sparse archives).
#   GP gives the most informative uncertainty.  Sinusoid is fastest but
#   assumes a fixed seasonal shape.  Spline makes the fewest assumptions.

println("\n── Figure 1: small n, all three methods ──")

r_sp  = disaggregate(Spline(), y_small, t1_small, t2_small)
r_sin = disaggregate(Sinusoid(), y_small, t1_small, t2_small)
kern = 12.0^2 * PeriodicKernel(r=[0.5]) * with_lengthscale(Matern52Kernel(), 3.0) +
       4.0^2 * with_lengthscale(Matern52Kernel(), 2.0)
r_gp  = disaggregate(GP(obs_noise = noise_std^2, kernel = kern),
                     y_small, t1_small, t2_small)

t_out_s = t_axis(r_sp)
sp_μ  = r_sp.signal.data;   sp_σ  = r_sp.std.data
sin_μ = r_sin.signal.data;  sin_σ = r_sin.std.data
gp_μ  = r_gp.signal.data;   gp_σ  = r_gp.std.data

# Build interval line-segments for the "input" panel.
# linesegments! expects a flat Vector of Point2f: [p1_start, p1_end, p2_start, …]
make_segs(t1v, t2v, yv) =
    vcat([Point2f[Point2f(decimal_year(t1v[i]), yv[i]),
                  Point2f(decimal_year(t2v[i]), yv[i])]
          for i in eachindex(yv)]...)

segs_s = make_segs(t1_small, t2_small, y_small)

fig1 = Figure(size = (1000, 720), fontsize = 12);

ax1a = Axis(fig1[1, 1]; xlabel = "Year", ylabel = "Signal",
    title = "Input: $n_small sparse intervals (≈ 3–8 months each)")
linesegments!(ax1a, segs_s; color = (:black, 0.4), linewidth = 2.5,
    label = "Interval averages (input)")
lines!(ax1a, t_decyear, signal; color = (:black, 0.2), linewidth = 1,
    label = "True signal")
axislegend(ax1a; position = :lt, labelsize = 11)

ax1b = Axis(fig1[1, 2]; xlabel = "Year", ylabel = "Signal",
    title = "Spline")
band!(ax1b, t_out_s, sp_μ .- 2sp_σ, sp_μ .+ 2sp_σ;
    color = (:steelblue, 0.2), label = "± 2σ")
lines!(ax1b, t_out_s, sp_μ; color = :steelblue, linewidth = 2.5,
    label = "Spline mean")
lines!(ax1b, t_decyear, signal; color = (:black, 0.2), linewidth = 1,
    label = "True signal")
axislegend(ax1b; position = :lt, labelsize = 11)

ax1c = Axis(fig1[2, 1]; xlabel = "Year", ylabel = "Signal",
    title = "Sinusoid — fastest; assumes seasonal shape")
band!(ax1c, t_out_s, sin_μ .- 2sin_σ, sin_μ .+ 2sin_σ;
    color = (:darkorange, 0.2), label = "± 2σ")
lines!(ax1c, t_out_s, sin_μ; color = :darkorange, linewidth = 2.5,
    label = "Sinusoid mean")
lines!(ax1c, t_decyear, signal; color = (:black, 0.2), linewidth = 1,
    label = "True signal")
axislegend(ax1c; position = :lt, labelsize = 11)

ax1d = Axis(fig1[2, 2]; xlabel = "Year", ylabel = "Signal",
    title = "GP — most meaningful uncertainty for sparse data")
band!(ax1d, t_out_s, gp_μ .- 2gp_σ, gp_μ .+ 2gp_σ;
    color = (:crimson, 0.2), label = "± 2σ")
lines!(ax1d, t_out_s, gp_μ; color = :crimson, linewidth = 2.5,
    label = "GP mean")
lines!(ax1d, t_decyear, signal; color = (:black, 0.2), linewidth = 1,
    label = "True signal")
axislegend(ax1d; position = :lt, labelsize = 11)

display(fig1)
save(joinpath(fig_dir, "fig1_small_n_three_methods.png"), fig1, px_per_unit = 2)
println("Saved fig1_small_n_three_methods.png")

# ─────────────────────────────────────────────────────────────────────────────
# Section 3 — Figure 2: Large n, all three methods (with timing)
# ─────────────────────────────────────────────────────────────────────────────
# When to use:
#   Dense short-interval data (in-situ sensors, reprocessed imagery).
#   With n=2000 all three methods recover the signal well.
#   Sinusoid is fastest (O(n) design matrix); GP is O(m³) not O(n³) so still
#   tractable; spline is accurate and flexible.

println("\n── Figure 2: large n ($n_large observations), timing ──")

print("  Spline:   "); @time r_sp_l  = disaggregate(Spline(smoothness = 1e-4),
                                                      y_large, t1_large, t2_large)
print("  Sinusoid: "); @time r_sin_l = disaggregate(Sinusoid(),
                                                      y_large, t1_large, t2_large)
print("  GP:       "); @time r_gp_l  = disaggregate(GP(obs_noise = noise_std^2),
                                                      y_large, t1_large, t2_large)

t_out_l  = t_axis(r_sp_l)
sp_μ_l   = r_sp_l.signal.data
sin_μ_l  = r_sin_l.signal.data
gp_μ_l   = r_gp_l.signal.data
gp_σ_l   = r_gp_l.std.data

fig2 = Figure(size = (1200, 380), fontsize = 12);

ax2a = Axis(fig2[1, 1]; xlabel = "Year", ylabel = "Signal",
    title = "Spline  (smoothness=1e-4, n=$n_large)");
lines!(ax2a, t_out_l, sp_μ_l;  color = :steelblue,  linewidth = 2, label = "Spline")
lines!(ax2a, t_decyear, signal; color = (:black, 0.2), linewidth = 1, label = "True signal")
axislegend(ax2a; position = :lt, labelsize = 10)

ax2b = Axis(fig2[1, 2]; xlabel = "Year", ylabel = "Signal",
    title = "Sinusoid  (fastest method, n=$n_large)")
lines!(ax2b, t_out_l, sin_μ_l; color = :darkorange,  linewidth = 2, label = "Sinusoid")
lines!(ax2b, t_decyear, signal; color = (:black, 0.2), linewidth = 1, label = "True signal")
axislegend(ax2b; position = :lt, labelsize = 10)

ax2c = Axis(fig2[1, 3]; xlabel = "Year", ylabel = "Signal",
    title = "GP  (O(m³) not O(n³), n=$n_large)")
band!(ax2c, t_out_l, gp_μ_l .- 2gp_σ_l, gp_μ_l .+ 2gp_σ_l;
    color = (:crimson, 0.15), label = "± 2σ")
lines!(ax2c, t_out_l, gp_μ_l;  color = :crimson,     linewidth = 2, label = "GP mean")
lines!(ax2c, t_decyear, signal; color = (:black, 0.2), linewidth = 1, label = "True signal")
axislegend(ax2c; position = :lt, labelsize = 10)

display(fig2)
save(joinpath(fig_dir, "fig2_large_n_timing.png"), fig2, px_per_unit = 2)
println("Saved fig2_large_n_timing.png")

# ─────────────────────────────────────────────────────────────────────────────
# Section 4 — Figure 3: Spline — effect of smoothness and tension
# ─────────────────────────────────────────────────────────────────────────────
# smoothness (λ): larger = smoother.  Too small → overfitting; too large →
#   over-smoothing.  Default 1e-3 works well for most cases.
# tension (τ): adds a first-derivative penalty alongside the curvature penalty.
#   tension≈0.5–1 suppresses oscillation; tension≈5–10 → piecewise-linear.
# penalty_order: 2 penalises curvature (∫x″²); 3 (default) penalises rate of
#   curvature change (∫x‴²); higher → polynomial extrapolation at boundaries.

println("\n── Figure 3: spline smoothness and tension kwargs ──")

smoothness_vals = [1e-5, 1e-3, 1e-1]
tension_vals    = [0.0, 1.0, 10.0]

# Row 1: vary smoothness at tension = 0
r_sp_row1 = [disaggregate(Spline(smoothness = s, tension = 0.0),
                          y_small, t1_small, t2_small)
             for s in smoothness_vals]
# Row 2: vary tension at smoothness = 1e-3
r_sp_row2 = [disaggregate(Spline(smoothness = 1e-3, tension = τ),
                          y_small, t1_small, t2_small)
             for τ in tension_vals]

fig3 = Figure(size = (1100, 600), fontsize = 11);

titles_r1 = ["smoothness=1e-5  (overfit)",
             "smoothness=1e-3  (default)",
             "smoothness=1e-1  (underfit)"]
titles_r2 = ["tension=0  (standard P-spline)",
             "tension=1  (moderate stiffening)",
             "tension=10  (near piecewise-linear)"]

for (col, r) in enumerate(r_sp_row1)
    ax = Axis(fig3[1, col]; xlabel = "Year", ylabel = "Signal",
              title = titles_r1[col])
    μ = r.signal.data; σ = r.std.data; t = t_axis(r)
    band!(ax, t, μ .- 2σ, μ .+ 2σ; color = (:steelblue, 0.2))
    lines!(ax, t, μ; color = :steelblue, linewidth = 2)
    lines!(ax, t_decyear, signal; color = (:black, 0.2), linewidth = 1)
end
for (col, r) in enumerate(r_sp_row2)
    ax = Axis(fig3[2, col]; xlabel = "Year", ylabel = "Signal",
              title = titles_r2[col])
    μ = r.signal.data; σ = r.std.data; t = t_axis(r)
    band!(ax, t, μ .- 2σ, μ .+ 2σ; color = (:purple, 0.2))
    lines!(ax, t, μ; color = :purple, linewidth = 2)
    lines!(ax, t_decyear, signal; color = (:black, 0.2), linewidth = 1)
end

display(fig3)
save(joinpath(fig_dir, "fig3_spline_kwargs.png"), fig3, px_per_unit = 2)
println("Saved fig3_spline_kwargs.png")

# ─────────────────────────────────────────────────────────────────────────────
# Section 5 — Figure 4: Sinusoid — fitted parameter decomposition
# ─────────────────────────────────────────────────────────────────────────────
# The sinusoid method fits x(t) = μ + β·(t−t̄) + γ(year) + A·sin(2πt) + B·cos(2πt)
# All components are returned in metadata(result):
#   :mean        → μ (overall mean)
#   :trend       → β (slope, units/year)
#   :amplitude   → √(A²+B²) (seasonal amplitude)
#   :phase       → seasonal peak time as fraction of year (0 = Jan 1)
#   :interannual → Dict{Int,Float64}  year → γ (anomaly relative to mean+trend)

println("\n── Figure 4: sinusoid parameter decomposition ──")

r_sin2 = disaggregate(Sinusoid(smoothness_interannual = 1e-2),
                       y_small, t1_small, t2_small)
md = DimensionalData.metadata(r_sin2)

t_out_s2  = t_axis(r_sin2)
sin2_μ    = r_sin2.signal.data
sin2_σ    = r_sin2.std.data

# Approximate trend reference as midpoint of output time axis
t_ref_approx = mean(t_out_s2)
trend_line   = md[:mean] .+ md[:trend] .* (t_out_s2 .- t_ref_approx)
amplitude    = md[:amplitude]

# Interannual anomalies bar chart
γ_dict  = md[:interannual]
yr_keys = sort(collect(keys(γ_dict)))
γ_vals  = [γ_dict[yr] for yr in yr_keys]

fig4 = Figure(size = (1000, 420), fontsize = 12)

ax4a = Axis(fig4[1, 1]; xlabel = "Year", ylabel = "Signal",
    title = "Sinusoid decomposition: trend ± seasonal envelope (n=$n_small)")
# ± seasonal envelope around the trend
band!(ax4a, t_out_s2, trend_line .- amplitude, trend_line .+ amplitude;
    color = (:darkorange, 0.12), label = "±amplitude ($(round(amplitude, digits=1)))")
lines!(ax4a, t_out_s2, trend_line; color = (:darkorange, 0.7), linewidth = 2,
    linestyle = :dash, label = "Trend line (μ + β·t)")
lines!(ax4a, t_out_s2, sin2_μ; color = :darkorange, linewidth = 2.5,
    label = "Full reconstruction")
lines!(ax4a, t_decyear, signal; color = (:black, 0.2), linewidth = 1,
    label = "True signal")
axislegend(ax4a; position = :lt, labelsize = 10)

# Annotate recovered parameters
ann = "μ = $(round(md[:mean], digits=2))\n" *
      "β = $(round(md[:trend], digits=2)) /yr\n" *
      "A = $(round(amplitude, digits=2))\n" *
      "peak ≈ $(round(md[:phase] * 12, digits=1)) month"
text!(ax4a, 2023.5, minimum(sin2_μ) + 1; text = ann, fontsize = 10)

ax4b = Axis(fig4[1, 2]; xlabel = "Year", ylabel = "Interannual anomaly γ",
    title = "Per-year anomalies from metadata[:interannual]")
barplot!(ax4b, Float64.(yr_keys) .+ 0.5, γ_vals;
    color = [v > 0 ? :steelblue : :tomato for v in γ_vals],
    gap = 0.1)
hlines!(ax4b, [0.0]; color = :black, linewidth = 1)

save(joinpath(fig_dir, "fig4_sinusoid_params.png"), fig4, px_per_unit = 2)
println("Saved fig4_sinusoid_params.png")

# ─────────────────────────────────────────────────────────────────────────────
# Section 6 — Figure 5: GP — kernel design
# ─────────────────────────────────────────────────────────────────────────────
# The kernel encodes your prior belief about the signal's smoothness and structure.
# Use `kernel = ...` to pass any KernelFunctions.jl kernel (sums and products
# are fully supported).
#
# k_default : Matérn-5/2, 2-month length-scale — smooth, non-periodic
# k_seasonal: PeriodicKernel × Matérn-5/2 encodes annual cycle
#             + long-scale Matérn-5/2 captures slow trend
# k_long    : Matérn-5/2 with 2-year length-scale — very smooth, trend-like

println("\n── Figure 5: GP kernel design ──")

k_default = with_lengthscale(Matern52Kernel(), 1/6)   # 2-month scale (package default)

k_seasonal = 12.0^2 * PeriodicKernel(r = [0.5]) * with_lengthscale(Matern52Kernel(), 3.0) +
              4.0^2 * with_lengthscale(Matern52Kernel(), 2.0)  # annual cycle + slow trend

k_long = with_lengthscale(Matern52Kernel(), 2.0)       # 2-year scale — very smooth

kernels = [k_default, k_seasonal, k_long]
kernel_names = ["Default: Matérn-5/2 (ℓ=2 months)",
                "Seasonal: PeriodicKernel × Matérn + slow Matérn",
                "Long-scale: Matérn-5/2 (ℓ=2 years)"]

r_gp_kernels = [disaggregate(GP(kernel = k, obs_noise = noise_std^2),
                             y_small, t1_small, t2_small)
                for k in kernels]

colors_gp = [:crimson, :darkgreen, :purple]

fig5 = Figure(size = (1100, 380), fontsize = 12)
for (col, (r, name, col_c)) in enumerate(zip(r_gp_kernels, kernel_names, colors_gp))
    ax = Axis(fig5[1, col]; xlabel = "Year", ylabel = "Signal", title = name)
    μ = r.signal.data; σ = r.std.data; t = t_axis(r)
    band!(ax, t, μ .- 2σ, μ .+ 2σ; color = (col_c, 0.18), label = "± 2σ")
    lines!(ax, t, μ; color = col_c, linewidth = 2.5, label = "GP mean")
    lines!(ax, t_decyear, signal; color = (:black, 0.2), linewidth = 1,
           label = "True signal")
    axislegend(ax; position = :lt, labelsize = 10)
end

save(joinpath(fig_dir, "fig5_gp_kernels.png"), fig5, px_per_unit = 2)
println("Saved fig5_gp_kernels.png")

# ─────────────────────────────────────────────────────────────────────────────
# Section 7 — Figure 6: L1 vs L2 robustness to blunders
# ─────────────────────────────────────────────────────────────────────────────
# loss_norm = :L2 (default): ordinary least squares — blunders pull the
#   reconstruction off truth.
# loss_norm = :L1: Iteratively Reweighted Least Squares (IRLS) — automatically
#   down-weights gross outliers.  Works with any method.

println("\n── Figure 6: L1 vs L2 blunder suppression ──")

# Inject 3 blunders ≈ 5× signal std at positions 4, 14, 24
y_blundered = copy(y_small)
blunder_idx = [3, round(Int, n_small ÷ 2), n_small - 2]
y_blundered[blunder_idx] .+= 5 * std(y_small)

r_l2 = disaggregate(Spline(), y_blundered, t1_small, t2_small;
                     loss_norm = :L2)
r_l1 = disaggregate(Spline(), y_blundered, t1_small, t2_small;
                     loss_norm = :L1)

t_bl   = t_axis(r_l2)
l2_μ   = r_l2.signal.data;  l2_σ  = r_l2.std.data
l1_μ   = r_l1.signal.data;  l1_σ  = r_l1.std.data

# Interval segments, colour-coded: red for blunders, grey for clean
seg_colors = [i in blunder_idx ? (:crimson, 0.8) : (:black, 0.3)
              for i in 1:n_small]

segs_bl       = make_segs(t1_small, t2_small, y_blundered)
segs_bl_clean = make_segs(t1_small[setdiff(1:n_small, blunder_idx)],
                           t2_small[setdiff(1:n_small, blunder_idx)],
                           y_blundered[setdiff(1:n_small, blunder_idx)])
segs_blunders = make_segs(t1_small[blunder_idx], t2_small[blunder_idx],
                           y_blundered[blunder_idx])

fig6 = Figure(size = (1000, 420), fontsize = 12)

ax6a = Axis(fig6[1, 1]; xlabel = "Year", ylabel = "Signal",
    title = "L2 loss — blunders (red) distort reconstruction")
linesegments!(ax6a, segs_bl_clean;  color = (:black, 0.3), linewidth = 2, label = "Clean intervals")
linesegments!(ax6a, segs_blunders;  color = (:crimson, 0.9), linewidth = 3, label = "Blunders (injected)")
band!(ax6a, t_bl, l2_μ .- 2l2_σ, l2_μ .+ 2l2_σ; color = (:steelblue, 0.2))
lines!(ax6a, t_bl, l2_μ; color = :steelblue, linewidth = 2.5, label = "L2 spline")
lines!(ax6a, t_decyear, signal; color = (:black, 0.2), linewidth = 1, label = "True signal")
axislegend(ax6a; position = :lt, labelsize = 11)

ax6b = Axis(fig6[1, 2]; xlabel = "Year", ylabel = "Signal",
    title = "L1 loss — blunders automatically down-weighted")
linesegments!(ax6b, segs_bl_clean;  color = (:black, 0.3), linewidth = 2, label = "Clean intervals")
linesegments!(ax6b, segs_blunders;  color = (:crimson, 0.9), linewidth = 3, label = "Blunders (injected)")
band!(ax6b, t_bl, l1_μ .- 2l1_σ, l1_μ .+ 2l1_σ; color = (:forestgreen, 0.2))
lines!(ax6b, t_bl, l1_μ; color = :forestgreen, linewidth = 2.5, label = "L1 spline")
lines!(ax6b, t_decyear, signal; color = (:black, 0.2), linewidth = 1, label = "True signal")
axislegend(ax6b; position = :lt, labelsize = 11)

save(joinpath(fig_dir, "fig6_l1_vs_l2.png"), fig6, px_per_unit = 2)
println("Saved fig6_l1_vs_l2.png")

# ─────────────────────────────────────────────────────────────────────────────
# Section 8 — Figure 7: Output grid options
# ─────────────────────────────────────────────────────────────────────────────
# output_period: any Dates.Period — controls the output time step.
#   Default Month(1) → 1st of each month.
#   Day(1) → daily output (GP keeps inducing points monthly; extra kriging step).
# output_start: anchors the grid.
#   Date(2020,1,15) with Month(1) step → 15th of every month.

println("\n── Figure 7: output grid options ──")

r_monthly = disaggregate(Spline(), y_large, t1_large, t2_large)
r_daily   = disaggregate(Spline(), y_large, t1_large, t2_large;
                          output_period = Day(1))
r_15th    = disaggregate(Spline(), y_large, t1_large, t2_large;
                          output_start  = Date(2020, 1, 15))

t_mon   = t_axis(r_monthly)
t_daily2 = t_axis(r_daily)
t_15th  = t_axis(r_15th)
μ_mon   = r_monthly.signal.data
μ_day   = r_daily.signal.data
μ_15th  = r_15th.signal.data

# Zoom to spring 2021 for clarity
zoom = (2021.0, 2021.35)
mask_m = zoom[1] .<= t_mon    .<= zoom[2]
mask_d = zoom[1] .<= t_daily2 .<= zoom[2]
mask_f = zoom[1] .<= t_15th   .<= zoom[2]

fig7 = Figure(size = (700, 380), fontsize = 12)
ax7 = Axis(fig7[1, 1]; xlabel = "Year", ylabel = "Signal",
    title = "Output grid options (3-month zoom, large dataset)")
lines!(ax7,   t_daily2[mask_d], μ_day[mask_d]; color = (:steelblue, 0.5),
    linewidth = 1, label = "Daily (output_period=Day(1))")
scatter!(ax7, t_mon[mask_m], μ_mon[mask_m]; color = :black,
    markersize = 8, marker = :circle,
    label = "Monthly 1st (default)")
scatter!(ax7, t_15th[mask_f], μ_15th[mask_f]; color = :crimson,
    markersize = 8, marker = :utriangle,
    label = "Monthly 15th (output_start=Date(2020,1,15))")
axislegend(ax7; position = :lt, labelsize = 10)

save(joinpath(fig_dir, "fig7_output_grid.png"), fig7, px_per_unit = 2)
println("Saved fig7_output_grid.png")

println("\nDone. All figures saved to $(fig_dir)/")
