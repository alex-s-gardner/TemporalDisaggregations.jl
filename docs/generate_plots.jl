using TemporalDisaggregations
using KernelFunctions
using CairoMakie
using DimensionalData: dims, Ti
using Statistics
using Dates
using Random

Random.seed!(7)

# ── Synthetic signal ──────────────────────────────────────────────────────────
t_daily = DateTime(2015, 1, 1):Day(1):DateTime(2020, 1, 1)
t_decyear = decimal_year.(t_daily)

seasonal_amp    = 15.0
interannual_amp = 6.0
interannual_T   = 3.0
trend_rate      = 4.0
noise_std       = 1.5

amp_t  = seasonal_amp .+ interannual_amp .* sin.(2π .* (t_decyear .- 2015.0) ./ interannual_T)
signal = amp_t .* sin.(2π .* t_decyear) .+ trend_rate .* (t_decyear .- 2015.0) .+ noise_std .* randn(length(t_decyear))

# ── Random overlapping intervals ──────────────────────────────────────────────
n        = 60
interval_length = 182
t_center = rand(t_daily, n)
t1 = max.(t_daily[1], t_center .- Dates.Day.(round.(Int, rand(n) .* interval_length)))
t2 = min.(t_daily[end], t_center .+ Dates.Day.(round.(Int, rand(n) .* interval_length)))
y        = [mean(signal[(t1[i] .<= t_daily) .& (t_daily .<= t2[i])]) for i in 1:n]

# ── Disaggregate ──────────────────────────────────────────────────────────────
r_spline = disaggregate(y, t1, t2; method = :spline,   smoothness = 1e-3)
r_sin    = disaggregate(y, t1, t2; method = :sinusoid, smoothness_interannual = 1e-2)

k = 15.0^2 * PeriodicKernel(r=[0.5]) * with_lengthscale(Matern52Kernel(), 3.0) +
     5.0^2 * with_lengthscale(Matern52Kernel(), 2.0) +
     3.0^2 * with_lengthscale(Matern32Kernel(), 1/12)
r_gp = disaggregate(y, t1, t2; method = :gp, kernel = k, obs_noise = noise_std^2, n_quad = 5)

# Extract plain Float64 vectors (avoids Makie/DimensionalData extension conflicts)
# Scalar indexing on a DimArray always returns a plain scalar.

sp_μ = r_spline.signal.data
sin_μ = r_sin.signal.data
gp_μ = r_gp.signal.data
gp_σ  = r_gp.std.data

# Monthly time axis from the shared output grid
t_output = decimal_year.(dims(r_spline, :Ti).val)

# ── Figure 1: overview of all three methods ───────────────────────────────────
fig1 = Figure(size = (900, 480), fontsize = 13);
ax1 = Axis(fig1[1, 1];
    xlabel = "Year", ylabel = "Signal",
    title  = "Temporal disaggregation: recover instantaneous signal from interval averages",
)

pt1 = Point2f.(decimal_year.(t1), y);
pt2 = Point2f.(decimal_year.(t2), y);
linesegments!(ax1, vcat(collect(zip(pt1, pt2))...);
    color = (:black, 0.35), linewidth = 2, label = "Interval averages (input)")
lines!(ax1, t_decyear, signal; color = (:black, 0.2), linewidth = 1, label = "True instantaneous signal")

band!(ax1, t_output, gp_μ .- 2 .* gp_σ, gp_μ .+ 2 .* gp_σ; color = (:crimson, 0.15))
lines!(ax1, t_output, gp_μ;  color = :crimson,   linewidth = 2.5, label = "GP (mean ± 2σ)")
lines!(ax1, t_output, sp_μ;  color = :steelblue, linewidth = 2.5, linestyle = :dash, label = "B-spline")
lines!(ax1, t_output, sin_μ; color = :darkorange, linewidth = 2.5, linestyle = :dot,  label = "Sinusoid")

axislegend(ax1; position = :lt, framevisible = true, labelsize = 12)
save("docs/images/overview.png", fig1, px_per_unit = 2)
println("Saved overview.png")

# ── Figure 2: problem / solution side-by-side ─────────────────────────────────
fig2 = Figure(size = (900, 420), fontsize = 13);

ax2a = Axis(fig2[1, 1]; xlabel = "Year", ylabel = "Signal",
    title = "Input: overlapping interval averages")
linesegments!(ax2a, vcat(collect(zip(pt1, pt2))...);
    color = (:steelblue, 0.6), linewidth = 2.5, label = "Interval averages")
lines!(ax2a, t_decyear, signal; color = (:black, 0.25), linewidth = 1, label = "True signal")
axislegend(ax2a; position = :lt, framevisible = true, labelsize = 11)

ax2b = Axis(fig2[1, 2]; xlabel = "Year", ylabel = "Signal",
    title = "Output: GP posterior mean ± 2σ")
band!(ax2b, t_output, gp_μ .- 2 .* gp_σ, gp_μ .+ 2 .* gp_σ;
    color = (:crimson, 0.2), label = "± 2σ")
lines!(ax2b, t_output, gp_μ; color = :crimson, linewidth = 2.5, label = "Posterior mean")
lines!(ax2b, t_decyear, signal; color = (:black, 0.25), linewidth = 1, label = "True signal")
axislegend(ax2b; position = :lt, framevisible = true, labelsize = 11)

save("docs/images/gp_detail.png", fig2, px_per_unit = 2)
println("Saved gp_detail.png")

# ── Also need tension-spline and sinusoid std ──────────────────────────────────
r_tension = disaggregate(y, t1, t2; method = :spline, smoothness = 1e-3, tension = 10.0)

sp_σ      = r_spline.signal.data  
sp_std    = r_spline.std.data
ten_μ     = r_tension.signal.data
ten_std   = r_tension.std.data
sin_μ     = r_sin.signal.data
sin_std   = r_sin.std.data

# ── Figure 3: B-spline detail ─────────────────────────────────────────────────
fig3 = Figure(size = (900, 420), fontsize = 13);

ax3a = Axis(fig3[1, 1]; xlabel = "Year", ylabel = "Signal",
    title = "Input: overlapping interval averages")
linesegments!(ax3a, vcat(collect(zip(pt1, pt2))...);
    color = (:steelblue, 0.6), linewidth = 2.5, label = "Interval averages")
lines!(ax3a, t_decyear, signal; color = (:black, 0.25), linewidth = 1, label = "True signal")
axislegend(ax3a; position = :lt, framevisible = true, labelsize = 11)

ax3b = Axis(fig3[1, 2]; xlabel = "Year", ylabel = "Signal",
    title = "Output: B-spline mean ± 2σ")
band!(ax3b, t_output, sp_μ .- 2 .* sp_std, sp_μ .+ 2 .* sp_std;
    color = (:steelblue, 0.2), label = "± 2σ")
lines!(ax3b, t_output, sp_μ; color = :steelblue, linewidth = 2.5, label = "B-spline mean")
lines!(ax3b, t_decyear, signal; color = (:black, 0.25), linewidth = 1, label = "True signal")
axislegend(ax3b; position = :lt, framevisible = true, labelsize = 11)

save("docs/images/spline_detail.png", fig3, px_per_unit = 2)
println("Saved spline_detail.png")

# ── Figure 4: Tension-spline detail ───────────────────────────────────────────
fig4 = Figure(size = (900, 420), fontsize = 13);

ax4a = Axis(fig4[1, 1]; xlabel = "Year", ylabel = "Signal",
    title = "Input: overlapping interval averages")
linesegments!(ax4a, vcat(collect(zip(pt1, pt2))...);
    color = (:purple, 0.6), linewidth = 2.5, label = "Interval averages")
lines!(ax4a, t_decyear, signal; color = (:black, 0.25), linewidth = 1, label = "True signal")
axislegend(ax4a; position = :lt, framevisible = true, labelsize = 11)

ax4b = Axis(fig4[1, 2]; xlabel = "Year", ylabel = "Signal",
    title = "Output: tension-spline mean ± 2σ  (tension = 10)")
band!(ax4b, t_output, ten_μ .- 2 .* ten_std, ten_μ .+ 2 .* ten_std;
    color = (:purple, 0.2), label = "± 2σ")
lines!(ax4b, t_output, ten_μ; color = :purple, linewidth = 2.5, label = "Tension-spline mean")
lines!(ax4b, t_decyear, signal; color = (:black, 0.25), linewidth = 1, label = "True signal")
axislegend(ax4b; position = :lt, framevisible = true, labelsize = 11)

save("docs/images/tension_spline_detail.png", fig4, px_per_unit = 2)
println("Saved tension_spline_detail.png")

# ── Figure 5: Sinusoid detail ─────────────────────────────────────────────────
fig5 = Figure(size = (900, 420), fontsize = 13);

ax5a = Axis(fig5[1, 1]; xlabel = "Year", ylabel = "Signal",
    title = "Input: overlapping interval averages")
linesegments!(ax5a, vcat(collect(zip(pt1, pt2))...);
    color = (:darkorange, 0.6), linewidth = 2.5, label = "Interval averages")
lines!(ax5a, t_decyear, signal; color = (:black, 0.25), linewidth = 1, label = "True signal")
axislegend(ax5a; position = :lt, framevisible = true, labelsize = 11)

ax5b = Axis(fig5[1, 2]; xlabel = "Year", ylabel = "Signal",
    title = "Output: sinusoid mean ± 2σ")
band!(ax5b, t_output, sin_μ .- 2 .* sin_std, sin_μ .+ 2 .* sin_std;
    color = (:darkorange, 0.2), label = "± 2σ")
lines!(ax5b, t_output, sin_μ; color = :darkorange, linewidth = 2.5, label = "Sinusoid mean")
lines!(ax5b, t_decyear, signal; color = (:black, 0.25), linewidth = 1, label = "True signal")
axislegend(ax5b; position = :lt, framevisible = true, labelsize = 11)

save("docs/images/sinusoid_detail.png", fig5, px_per_unit = 2)
println("Saved sinusoid_detail.png")

# ── Figure 6: lines(result[:signal]) quick-start demo ────────────────────────
# Mirrors the one-liner shown in the README Quick Start section.
# Uses decimal-year x-axis since DimensionalData's Ti dimension plots that way.
fig6 = Figure(size = (700, 300), fontsize = 13);
ax6 = Axis(fig6[1, 1]; xlabel = "Year", ylabel = "Signal")
lines!(ax6, t_output, gp_μ; color = :steelblue, linewidth = 2)
save("docs/images/lines_signal.png", fig6, px_per_unit = 2)
println("Saved lines_signal.png")

# ── Figure 7: lines(result[:signal]) README quick-start demo ─────────────────
# Reproduces what `lines(result[:signal])` produces: Date x-axis, signal y-axis.
t_dates = dims(r_gp, :Ti).val
fig7 = Figure(size = (700, 300), fontsize = 13);
ax7 = Axis(fig7[1, 1]; xlabel = "Date", ylabel = "Signal")
lines!(ax7, t_dates, gp_μ; color = :steelblue, linewidth = 2)
save("docs/images/quickstart_lines_signal.png", fig7, px_per_unit = 2)
println("Saved quickstart_lines_signal.png")
