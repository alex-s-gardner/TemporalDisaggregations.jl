# benchmarks/benchmark.jl — timing benchmarks for all three disaggregate methods
#
# Run from the repository root:
#   julia --project=. benchmarks/benchmark.jl
#
# Sizes tested: 1e4, 1e5, 1e6 observations (n), 20-year span, output_period=Week(1).
# Memory estimates (20-year span, weekly output):
#   Spline:   n × n_basis (≈240)  × 8 B  → 19 MB / 192 MB / 1.9 GB
#   Sinusoid: n × n_params (≈24)  × 8 B  →  2 MB /  19 MB / 192 MB
#   GP:       n × n_ind   (≈2430) × 8 B  → 195 MB / 1.95 GB / 19.5 GB
#     (n_ind large because inducing period ≈ Day(3) = half of Week(1))
# Methods that would exceed MEM_LIMIT_GB are skipped automatically.

using TemporalDisaggregations
using KernelFunctions
using Dates
using Statistics
using Random
using LinearAlgebra
using Printf

const MEM_LIMIT_GB = 8.0          # skip a run if its dominant matrix > this
const OUT_PERIOD   = Week(1)

# ── Kernel used for GP ────────────────────────────────────────────────────────
const GP_KERNEL = 15.0^2 * PeriodicKernel(r=[0.5]) *
                  with_lengthscale(Matern52Kernel(), 3.0) +
                  5.0^2 * with_lengthscale(Matern52Kernel(), 2.0)

# ── Generate n synthetic interval-average observations over a 20-year span ───
function make_data(n::Int; rng = Random.default_rng())
    t_start = Date(2000, 1, 1)
    t_end   = Date(2020, 1, 1)
    span    = (t_end - t_start).value        # total days in span

    # Random interval centres uniformly over the span
    centers = t_start .+ Day.(rand(rng, 0:span-1, n))

    # Half-lengths: 1–30 days
    half = rand(rng, 1:30, n)
    t1   = max.(t_start, centers .- Day.(half))
    t2   = min.(t_end,   centers .+ Day.(half))

    # Synthetic signal: annual sinusoid + linear trend
    tdy = (yeardecimal.(t1) .+ yeardecimal.(t2)) ./ 2
    y   = 10.0 .* sin.(2π .* tdy) .+ 3.0 .* (tdy .- 2000.0)

    return y, t1, t2
end

# ── Memory estimate helpers (bytes of dominant matrix) ────────────────────────
# 20-year span, weekly output → inducing period ≈ Day(3)
spline_mem(n;   n_basis  = 240)  = n * n_basis  * 8
sinusoid_mem(n; n_params = 24)   = n * n_params * 8
gp_mem(n;       n_ind    = 2430) = n * n_ind    * 8

# ── Single timed run returning (elapsed_s, alloc_bytes) ──────────────────────
function time_run(f)
    stats = @timed f()
    return stats.time, stats.bytes
end

# ── Pretty-print helpers ──────────────────────────────────────────────────────
fmt_time(s)  = s < 1e-3 ? @sprintf("%.1f μs", s * 1e6) :
               s < 1.0   ? @sprintf("%.1f ms", s * 1e3) :
                            @sprintf("%.2f s",  s)

fmt_bytes(b) = b < 1024^2 ? @sprintf("%.1f KB", b / 1024) :
               b < 1024^3 ? @sprintf("%.1f MB", b / 1024^2) :
                             @sprintf("%.2f GB", b / 1024^3)

# ── Benchmark runner ──────────────────────────────────────────────────────────
function run_benchmarks()
    sizes = [10_000, 100_000, 1_000_000]
    nruns = [3, 2, 1]

    cpu_info = Sys.cpu_info()
    cpu_name = isempty(cpu_info) ? "unknown" : cpu_info[1].model

    println("\n", "="^72)
    println(" TemporalDisaggregations.jl — timing benchmark")
    println(" Julia ", VERSION, "  threads=", Threads.nthreads())
    println(" CPU: ", cpu_name, "  (", length(cpu_info), " logical cores)")
    println(" RAM: ", round(Sys.total_memory() / 1024^3, digits = 1), " GB")
    println(" Span: 20 years (2000–2020)  output_period=Week(1)")
    println("="^72)

    rng = MersenneTwister(42)

    for (ni, n) in enumerate(sizes)
        println("\n── n = ", lpad(string(n), 10, '_'), " observations ──────────────────────────")

        print("  Generating data ... ")
        flush(stdout)
        y, t1, t2 = make_data(n; rng)
        t_gen = @elapsed make_data(n; rng = MersenneTwister(42))   # timed dry-run
        println(fmt_time(t_gen))

        nr = nruns[ni]

        # ── Spline ──────────────────────────────────────────────────────────
        mem_est = spline_mem(n)
        if mem_est / 1024^3 > MEM_LIMIT_GB
            @printf("  Spline   : SKIPPED (C_norm ≈ %.1f GB > %.0f GB limit)\n",
                    mem_est / 1024^3, MEM_LIMIT_GB)
        else
            print("  Spline   : warming up ... "); flush(stdout)
            try
                disaggregate(Spline(smoothness = 1e-3), y, t1, t2;
                             output_period = OUT_PERIOD)
                times = Float64[]; allocs = Float64[]
                for _ in 1:nr
                    t, b = time_run(() -> disaggregate(Spline(smoothness = 1e-3), y, t1, t2;
                                                       output_period = OUT_PERIOD))
                    push!(times, t); push!(allocs, b)
                end
                @printf("%d runs → %s  (alloc %s)\n", nr,
                        fmt_time(minimum(times)), fmt_bytes(minimum(allocs)))
            catch e
                println("ERROR: ", e)
            end
        end

        # ── Sinusoid ─────────────────────────────────────────────────────────
        mem_est = sinusoid_mem(n)
        if mem_est / 1024^3 > MEM_LIMIT_GB
            @printf("  Sinusoid : SKIPPED (D ≈ %.1f GB > %.0f GB limit)\n",
                    mem_est / 1024^3, MEM_LIMIT_GB)
        else
            print("  Sinusoid : warming up ... "); flush(stdout)
            try
                disaggregate(Sinusoid(), y, t1, t2; output_period = OUT_PERIOD)
                times = Float64[]; allocs = Float64[]
                for _ in 1:nr
                    t, b = time_run(() -> disaggregate(Sinusoid(), y, t1, t2;
                                                       output_period = OUT_PERIOD))
                    push!(times, t); push!(allocs, b)
                end
                @printf("%d runs → %s  (alloc %s)\n", nr,
                        fmt_time(minimum(times)), fmt_bytes(minimum(allocs)))
            catch e
                println("ERROR: ", e)
            end
        end

        # ── GP ───────────────────────────────────────────────────────────────
        mem_est = gp_mem(n)
        if mem_est / 1024^3 > MEM_LIMIT_GB
            @printf("  GP       : SKIPPED (C ≈ %.1f GB > %.0f GB limit)\n",
                    mem_est / 1024^3, MEM_LIMIT_GB)
        else
            print("  GP       : warming up ... "); flush(stdout)
            try
                disaggregate(GP(kernel = GP_KERNEL, obs_noise = 4.0, n_quad = 5),
                             y, t1, t2; output_period = OUT_PERIOD)
                times = Float64[]; allocs = Float64[]
                for _ in 1:nr
                    t, b = time_run(() ->
                        disaggregate(GP(kernel = GP_KERNEL, obs_noise = 4.0, n_quad = 5),
                                     y, t1, t2; output_period = OUT_PERIOD))
                    push!(times, t); push!(allocs, b)
                end
                @printf("%d runs → %s  (alloc %s)\n", nr,
                        fmt_time(minimum(times)), fmt_bytes(minimum(allocs)))
            catch e
                println("ERROR: ", e)
            end
        end

    end

    println("\n", "="^72)
    println(" Notes:")
    println("   • Time = minimum over nruns (best-of for stable estimate)")
    println("   • alloc = total bytes allocated (GC pressure, not peak RSS)")
    println("   • Memory limit: $(MEM_LIMIT_GB) GB  (dominant input matrix; edit to override)")
    println("   • Spline  : O(n·n_basis + n_basis³)  n_basis ≈ 12·years  ≈ 240")
    println("   • Sinusoid: O(n·n_params)             n_params = 2+years+2 ≈ 24")
    println("   • GP      : O(n·n_ind·n_quad + n_ind³) n_ind ≈ 2·52·years ≈ 2430")
    println("   • GP builds C with Threads.@threads — run with --threads=auto")
    println("="^72, "\n")
end

run_benchmarks()
