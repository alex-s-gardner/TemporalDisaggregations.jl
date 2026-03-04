
#begin
# Example: GP-based disaggregation of interval-averaged data
#
# Compares the B-spline disaggregation (disaggregation) with the GP-based version
# (disaggregation_gp). Both recover an instantaneous signal from random overlapping
# interval averages mimicking ITS_LIVE image-pair velocity measurements.
begin
    using TemporalDisaggregations
    using KernelFunctions # kernel definitions
    using CairoMakie
    using DimensionalData: dims, Ti
    using Statistics
    using Dates
end

# ── 1. Synthetic instantaneous time series ────────────────────────────────────
# Signal components (all in m/yr):
#   • Long-term trend          — linear acceleration over the record
#   • Annual cycle             — fixed-amplitude seasonal swing
#   • Inter-annual modulation  — the seasonal amplitude itself waxes and wanes
#                                on a ~3-year cycle (realistic for glacier dynamics)
#   • Observation noise        — white noise added to each measurement
begin
    t_daily = DateTime(2015, 1, 1):Day(1):DateTime(2020, 1, 1)
    t       = [year(dt) + (dayofyear(dt) - 1) / (isleapyear(year(dt)) ? 366 : 365)
               for dt in t_daily]            # decimal years, for readability

    trend_rate      = 5.0                     # m/yr per year (linear speedup)
    seasonal_amp    = 15.0                    # base seasonal amplitude (m/yr)
    interannual_amp = 8.0                     # peak-to-peak modulation of seasonal amp (m/yr)
    interannual_T   = 3.0                     # period of amplitude modulation (years)
    noise_std       = 2.0                     # white noise std dev (m/yr)

    # Seasonal amplitude that varies inter-annually
    amp_t = seasonal_amp .+ interannual_amp .* sin.(2π .* (t .- 2015.0) ./ interannual_T)

    signal = (amp_t .* sin.(2π .* t)               # annual cycle with varying amplitude
        .+ trend_rate .* (t .- 2015.0)             # long-term trend
        .+ noise_std .* randn(length(t)))          # noise

end

# ── 2. Aggregate into random overlapping intervals ────────────────────────────
begin
    n        = 50
    sample_radius = 0.5;
    t_min    = t[1]
    t_max    = t[end]
    t_center = t_min .+ rand(n) .* (t_max - t_min)
    t1_dec   = max.(t_min, t_center .- rand(n) .* sample_radius)
    t2_dec   = min.(t_max, t_center .+ rand(n) .* sample_radius)

    y = [mean(signal[(t1_dec[i] .<= t) .& (t .<= t2_dec[i])]) for i in 1:n]

    dec_to_date(t) = let yr = floor(Int, t)
        Dates.Date(yr, 1, 1) + Dates.Day(floor(Int, (t - yr) * (isleapyear(yr) ? 366 : 365)))
    end
    t1 = dec_to_date.(t1_dec)
    t2 = dec_to_date.(t2_dec)
end
# ── 2b. Inject large, infrequent blunders ────────────────────────────────────
# Mimics gross errors in ITS_LIVE image-pair matching (e.g. mis-correlation,
# cloud/shadow contamination, or bad co-registration).
begin
    blunder_prob  = 0.01                        # ~1 % of observations are blunders
    blunder_scale = 5.0                           # blunder magnitude relative to signal std
    blunder_mask  = rand(n) .< blunder_prob       # which observations are corrupted
    blunder_sign  = rand([-1.0, 1.0], n)         # random direction
    y_blundered   = copy(y)
    y_blundered[blunder_mask] .+= blunder_sign[blunder_mask] .* blunder_scale .* std(y)

    println("Injected $(sum(blunder_mask)) blunders out of $n observations")
end


# ── 3. B-spline disaggregation ────────────────────────────────────────────────
@time result_bspline = disaggregate(y_blundered, t1, t2;
    method            = :spline,
    smoothness        = 1e-3,
    loss_norm         = :L1,
)

# ── 3b. Tension + L1 spline disaggregation ────────────────────────────────────
@time result_tension = disaggregate(y_blundered, t1, t2;
    method            = :spline,
    smoothness        = 1e-3,
    tension           = 1e1,
    loss_norm         = :L1,
)

# ── 4. Sinusoid disaggregation ────────────────────────────────────────────────
@time result_sin = disaggregate(y_blundered, t1, t2;
    method                 = :sinusoid,
    smoothness_interannual = 1e-2,
    loss_norm              = :L1,
)

# ── 5. GP disaggregation ──────────────────────────────────────────────────────
k_seasonal  = 15.0^2 * PeriodicKernel(r=[0.5]) * with_lengthscale(Matern52Kernel(), 3.0)
k_trend     = 5.0^2  * with_lengthscale(Matern52Kernel(), 2.0)
k_shortterm = 3.0^2  * with_lengthscale(Matern32Kernel(), 1/12)
k           = k_seasonal + k_trend + k_shortterm

@time result_gp = disaggregate(y_blundered, t1, t2;
    method    = :gp,
    kernel    = k,
    obs_noise = noise_std^2,
    loss_norm = :L1,
    n_quad    = 5,
)

# ── 5b. Daily output (demonstrates output_step) ───────────────────────────────
@time result_daily = disaggregate(y_blundered, t1, t2;
    method      = :spline,
    smoothness  = 1e-3,
    loss_norm   = :L1,
    output_step = Day(1),
)

# ── 6. Plot ───────────────────────────────────────────────────────────────────
begin
    fig = Figure(size = (1100, 600))
    ax  = Axis(fig[1, 1];
        xlabel = "Year",
        ylabel = "Velocity (m/yr)",
        title  = "Disaggregation: GP vs B-spline vs Sinusoid",
    )

    # Clean interval averages (light)
    pt1_clean = Point2f.(t1_dec[.!blunder_mask], y[.!blunder_mask])
    pt2_clean = Point2f.(t2_dec[.!blunder_mask], y[.!blunder_mask])
    segs_clean = vcat(collect(zip(pt1_clean, pt2_clean))...)
    linesegments!(ax, segs_clean; color = (:black, 0.5), linewidth = 1.5,
                  label = "Aggregated")

    # Blundered interval averages (red, prominent)
    if any(blunder_mask)
        pt1_bad = Point2f.(t1_dec[blunder_mask], y_blundered[blunder_mask])
        pt2_bad = Point2f.(t2_dec[blunder_mask], y_blundered[blunder_mask])
        segs_bad = vcat(collect(zip(pt1_bad, pt2_bad))...)
        linesegments!(ax, segs_bad; color = (:red, 0.9), linewidth = 2.5,
                    label = "Blunders")
    end

    # True instantaneous signal
    lines!(ax, t, signal;
           color = (:black, 0.35), linewidth = 1, label = "Instantaneous")

    # Shared time axis for all monthly results (same Ti dimension)
    t_monthly = [year(d) + (month(d) - 1) / 12.0 for d in dims(result_bspline.values, Ti)]

    # Daily output has its own finer Ti dimension
    t_daily_out = [year(d) + (dayofyear(d) - 1) / (isleapyear(year(d)) ? 366.0 : 365.0)
                   for d in dims(result_daily.values, Ti)]

    # B-spline reconstruction
    lines!(ax, t_monthly, Array(result_bspline.values);
           color = :steelblue, linewidth = 3, linestyle = :dash,
           label = "B-spline")

    # Tension spline reconstruction
    lines!(ax, t_monthly, Array(result_tension.values);
           color = :purple, linewidth = 3, linestyle = :dashdot,
           label = "Tension-spline")

    # Sinusoid reconstruction
    lines!(ax, t_monthly, Array(result_sin.values);
           color = :darkorange, linewidth = 2, linestyle = :dot,
           label = "Sinusoid")

    # Daily spline reconstruction (thin overlay to show sub-monthly detail)
    lines!(ax, t_daily_out, Array(result_daily.values);
           color = (:steelblue, 0.4), linewidth = 1, label = "B-spline (daily)")

    # GP reconstruction: posterior mean ± 2σ band
    band!(ax, t_monthly,
          Array(result_gp.values) .- 2 .* Array(result_gp.std),
          Array(result_gp.values) .+ 2 .* Array(result_gp.std);
          color = (:crimson, 0.15))
    lines!(ax, t_monthly, Array(result_gp.values);
           color = :crimson, linewidth = 2, label = "GP mean ± 2σ")

       # ylims!(ax, extrema(ts.data))
    axislegend(ax; position = :lt, framevisible = true)
    display(fig)
end end
