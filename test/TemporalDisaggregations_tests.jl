# ─────────────────────────────────────────────────────────────────────────────
# Shared setup: helper functions and common imports available to all testitems.
# ─────────────────────────────────────────────────────────────────────────────
@testsetup module TestHelpers
    using TemporalDisaggregations
    using Dates
    using Statistics: mean

    const TD = TemporalDisaggregations

    """Build n non-overlapping monthly Date intervals starting from t_start."""
    function make_monthly_intervals(t_start::Date, n::Int)
        t1 = [t_start + Month(i - 1) for i in 1:n]
        t2 = [t_start + Month(i)     for i in 1:n]
        return t1, t2
    end

    """Exact average of f over [t1, t2] via quadrature (for reference)."""
    function exact_average(f, t1::Float64, t2::Float64; npts=1000)
        ts = range(t1, t2; length=npts)
        return mean(f.(ts))
    end

    export TD, make_monthly_intervals, exact_average
end

# ─────────────────────────────────────────────────────────────────────────────
@testitem "Helper: _yeardecimal" setup=[TestHelpers] begin
    using Dates
    @test TD.yeardecimal(Date(2020, 1, 1)) == 2020.0
    @test TD.yeardecimal(Date(2000, 1, 1)) == 2000.0

    d_mid    = Date(2019, 7, 2)
    expected = 2019.0 + 182.0 / 365.0
    @test TD.yeardecimal(d_mid) ≈ expected atol=1e-12

    d_leap        = Date(2020, 7, 2)
    expected_leap = 2020.0 + 183.0 / 366.0
    @test TD.yeardecimal(d_leap) ≈ expected_leap atol=1e-12

    @test TD.yeardecimal(DateTime(2021, 1, 1, 0, 0, 0)) == 2021.0
end

# ─────────────────────────────────────────────────────────────────────────────
@testitem "Helper: _date_grid" setup=[TestHelpers] begin
    using Dates
    dates, times = TD._date_grid(Date(2020, 1, 1), Date(2021, 1, 1), Day(1))
    @test first(dates) == Date(2020, 1, 1)
    @test length(dates) > 300
    @test issorted(dates)
    @test all(isfinite, times)

    dates_m, _ = TD._date_grid(Date(2020, 1, 1), Date(2021, 1, 1), Month(1))
    @test first(dates_m) == Date(2020, 1, 1)
    @test last(dates_m)  == Date(2021, 1, 1)
    @test length(dates_m) == 13

    dates_os, _ = TD._date_grid(Date(2020, 5, 15), Date(2021, 1, 1), Month(1))
    @test first(dates_os) == Date(2020, 5, 15)
    @test all(d -> day(d) == 15, dates_os)

    dates_w, _ = TD._date_grid(Date(2020, 3, 1), Date(2021, 1, 1), Week(2))
    @test first(dates_w) == Date(2020, 3, 1)
end

# ─────────────────────────────────────────────────────────────────────────────
@testitem "Helper: _half_period" setup=[TestHelpers] begin
    using Dates
    @test TD._half_period(Year(1))   == Month(6)
    @test TD._half_period(Year(2))   == Month(12)
    @test TD._half_period(Month(6))  == Month(3)
    @test TD._half_period(Month(2))  == Month(1)
    @test TD._half_period(Month(1))  == Week(2)
    @test TD._half_period(Week(4))   == Day(14)
    @test TD._half_period(Week(1))   == Day(4)
    @test TD._half_period(Day(4))    == Day(2)
    @test TD._half_period(Day(1))    == Day(1)
    @test TD._half_period(Hour(1))   == Day(1)
end

# ─────────────────────────────────────────────────────────────────────────────
@testitem "Helper: _difference_matrix" setup=[TestHelpers] begin
    D1 = TD._difference_matrix(5, 1)
    @test size(D1) == (4, 5)
    @test D1 * ones(5) ≈ zeros(4) atol=1e-14
    @test D1[1, 1] ≈ -1.0
    @test D1[1, 2] ≈  1.0
    @test D1[2, 2] ≈ -1.0
    @test D1[2, 3] ≈  1.0

    D2 = TD._difference_matrix(6, 2)
    @test size(D2) == (4, 6)
    @test D2[1, 1] ≈  1.0
    @test D2[1, 2] ≈ -2.0
    @test D2[1, 3] ≈  1.0

    D3 = TD._difference_matrix(7, 3)
    @test D3[1, 1] ≈ -1.0
    @test D3[1, 2] ≈  3.0
    @test D3[1, 3] ≈ -3.0
    @test D3[1, 4] ≈  1.0

    @test_throws ArgumentError TD._difference_matrix(5, 0)
    @test_throws ArgumentError TD._difference_matrix(5, 5)
end

# ─────────────────────────────────────────────────────────────────────────────
@testitem "Helper: _irls_weights and _irls_converged" setup=[TestHelpers] begin
    r = [1.0, 2.0, 0.0, -3.0]
    ε = 1e-6
    w = TD._irls_weights(r, ε)
    @test length(w) == 4
    @test all(w .> 0)
    @test w[1] ≈ 1.0 / (1.0 + ε)
    @test w[3] ≈ 1.0 / ε

    x = [1.0, 2.0, 3.0]
    @test TD._irls_converged(x, x)
    @test !TD._irls_converged(x .+ 1.0, x)
end

# ─────────────────────────────────────────────────────────────────────────────
@testitem "Struct construction and defaults" setup=[TestHelpers] begin
    using TemporalDisaggregations

    s = Spline()
    @test s.smoothness == 1e-1
    @test s.n_knots === nothing
    @test s.penalty_order == 3
    @test s.tension == 0.0

    sin_m = Sinusoid()
    @test sin_m.smoothness_interannual == 1e-2

    gp_m = GP()
    @test gp_m.obs_noise == 1.0
    @test gp_m.n_quad == 5

    @test Spline()   isa DisaggregationMethod
    @test Sinusoid() isa DisaggregationMethod
    @test GP()       isa DisaggregationMethod

    @test_throws MethodError Spline(obs_noise=1.0)
    @test_throws MethodError GP(smoothness=1e-3)
end

# ─────────────────────────────────────────────────────────────────────────────
@testitem "Input validation" setup=[TestHelpers] begin
    using TemporalDisaggregations
    using Dates

    t1 = [Date(2020, 1, 1), Date(2020, 2, 1)]
    t2 = [Date(2020, 2, 1), Date(2020, 3, 1)]
    y  = [1.0, 2.0]

    @test_throws DimensionMismatch disaggregate(Spline(), [1.0], t1, t2)
    @test_throws DimensionMismatch disaggregate(Spline(), y, [Date(2020,1,1)], t2)
    @test_throws DimensionMismatch disaggregate(Spline(), y, t1, [Date(2020,2,1)])

    t2_bad = [Date(2020, 1, 1), Date(2020, 3, 1)]
    @test_throws ArgumentError disaggregate(Spline(), y, t1, t2_bad)
    t2_rev = [Date(2019, 12, 1), Date(2020, 3, 1)]
    @test_throws ArgumentError disaggregate(Spline(), y, t1, t2_rev)

    @test_throws ArgumentError disaggregate(Spline(smoothness=-1.0), y, t1, t2)
    @test_throws ArgumentError disaggregate(GP(obs_noise=-1.0), y, t1, t2)
end

# ─────────────────────────────────────────────────────────────────────────────
@testitem "Spline method: basics" setup=[TestHelpers] begin
    using TemporalDisaggregations
    using Dates
    using DimensionalData: DimStack, DimArray, Ti, dims, hasdim
    using Statistics: cor, mean

    @testset "Constant signal recovery" begin
        t1, t2 = make_monthly_intervals(Date(2020, 1, 1), 24)
        y = fill(5.0, 24)
        result = disaggregate(Spline(smoothness=1e-8), y, t1, t2)
        @test result isa DimStack
        @test result.signal isa DimArray
        @test result.std    isa DimArray
        @test hasdim(result.signal, Ti)
        @test length(result.signal) == length(result.std)
        @test mean(result.signal) ≈ 5.0 atol=0.1
        @test all(result.std.data .>= 0)
    end

    @testset "L1 vs L2 agree on clean data" begin
        t1, t2 = make_monthly_intervals(Date(2020, 1, 1), 24)
        y = [3.0 + sin(2π * (i / 12.0)) for i in 1:24]
        r_l2 = disaggregate(Spline(), y, t1, t2; loss_norm = :L2)
        r_l1 = disaggregate(Spline(), y, t1, t2; loss_norm = :L1)
        @test cor(r_l2.signal.data, r_l1.signal.data) > 0.99
    end

    @testset "output_period = Day(1) produces finer grid" begin
        t1, t2 = make_monthly_intervals(Date(2020, 1, 1), 12)
        y = ones(12)
        r_monthly = disaggregate(Spline(), y, t1, t2)
        r_daily   = disaggregate(Spline(), y, t1, t2; output_period = Day(1))
        @test length(r_daily.signal) > length(r_monthly.signal)
        @test collect(dims(r_daily.signal, Ti))[1] isa Date
    end

    @testset "tension > 0 returns valid result" begin
        t1, t2 = make_monthly_intervals(Date(2020, 1, 1), 24)
        y = [sin(2π * i / 12) for i in 1:24]
        r_notension = disaggregate(Spline(tension=0.0), y, t1, t2)
        r_tension   = disaggregate(Spline(tension=5.0), y, t1, t2)
        @test length(r_tension.signal) == length(r_notension.signal)
        @test all(isfinite, r_tension.signal)
        @test maximum(r_tension.signal) - minimum(r_tension.signal) <=
              maximum(r_notension.signal) - minimum(r_notension.signal) + 0.5
    end
end

# ─────────────────────────────────────────────────────────────────────────────
@testitem "Sinusoid method: basics" setup=[TestHelpers] begin
    using TemporalDisaggregations
    using Dates
    using DimensionalData: DimStack, DimArray, Ti, hasdim, metadata
    using Statistics: cor

    @testset "Recover known amplitude and phase" begin
        A_true, B_true = 3.0, 4.0
        amp_true   = sqrt(A_true^2 + B_true^2)
        phase_true = mod(atan(B_true, A_true) / (2π), 1.0)

        t1_dates, t2_dates = make_monthly_intervals(Date(2018, 1, 1), 36)
        t1_dec = TD.yeardecimal.(t1_dates)
        t2_dec = TD.yeardecimal.(t2_dates)

        y = [A_true * TD._interval_sin_integral(t1_dec[i], t2_dec[i]) +
             B_true * TD._interval_cos_integral(t1_dec[i], t2_dec[i])
             for i in eachindex(t1_dec)]

        result = disaggregate(Sinusoid(smoothness_interannual=1e-6),
                              y, t1_dates, t2_dates)
        @test metadata(result)[:amplitude] ≈ amp_true   atol=0.05
        @test metadata(result)[:phase]     ≈ phase_true atol=0.05
        @test metadata(result)[:trend]     ≈ 0.0        atol=0.05
    end

    @testset "Inter-annual dict has correct year keys" begin
        t1, t2 = make_monthly_intervals(Date(2019, 6, 1), 24)
        y = ones(24)
        result = disaggregate(Sinusoid(), y, t1, t2)
        @test metadata(result)[:interannual] isa Dict
        for yr in [2019, 2020, 2021]
            @test haskey(metadata(result)[:interannual], yr)
        end
    end

    @testset "L1 vs L2 agree on clean data" begin
        t1, t2 = make_monthly_intervals(Date(2020, 1, 1), 24)
        y = [2.0 + sin(2π * i / 12) for i in 1:24]
        r_l2 = disaggregate(Sinusoid(), y, t1, t2; loss_norm = :L2)
        r_l1 = disaggregate(Sinusoid(), y, t1, t2; loss_norm = :L1)
        @test cor(r_l2.signal.data, r_l1.signal.data) > 0.99
    end

    @testset "output_period = Day(1)" begin
        t1, t2 = make_monthly_intervals(Date(2020, 1, 1), 12)
        y = ones(12)
        r_monthly = disaggregate(Sinusoid(), y, t1, t2)
        r_daily   = disaggregate(Sinusoid(), y, t1, t2; output_period = Day(1))
        @test length(r_daily.signal) > length(r_monthly.signal)
        @test all(isfinite, r_daily.signal)
    end

    @testset "Return fields present" begin
        t1, t2 = make_monthly_intervals(Date(2020, 1, 1), 12)
        result = disaggregate(Sinusoid(), ones(12), t1, t2)
        @test result isa DimStack
        @test result.signal isa DimArray
        @test result.std    isa DimArray
        @test hasdim(result.signal, Ti)
        @test haskey(metadata(result), :mean)
        @test haskey(metadata(result), :trend)
        @test haskey(metadata(result), :amplitude)
        @test haskey(metadata(result), :phase)
        @test haskey(metadata(result), :interannual)
        @test metadata(result)[:amplitude] >= 0.0
        @test 0.0 <= metadata(result)[:phase] <= 1.0
        @test all(result.std.data .>= 0)
    end
end

# ─────────────────────────────────────────────────────────────────────────────
@testitem "GP method: basics" setup=[TestHelpers] begin
    using TemporalDisaggregations
    using Dates
    using DimensionalData: DimStack, DimArray, Ti, hasdim
    using Statistics: mean

    @testset "Returns DimStack with std ≥ 0" begin
        t1, t2 = make_monthly_intervals(Date(2020, 1, 1), 24)
        y = [sin(2π * i / 12) for i in 1:24]
        result = disaggregate(GP(obs_noise=0.1), y, t1, t2)
        @test result isa DimStack
        @test result.signal isa DimArray
        @test result.std    isa DimArray
        @test hasdim(result.signal, Ti)
        @test length(result.signal) == length(result.std)
        @test all(result.std.data .>= 0.0)
        @test all(isfinite, result.signal)
        @test all(isfinite, result.std)
    end

    @testset "output_period = Day(1) uses daily inducing grid" begin
        t1, t2 = make_monthly_intervals(Date(2020, 1, 1), 12)
        y = ones(12)
        r_monthly = disaggregate(GP(obs_noise=0.1), y, t1, t2)
        r_daily   = disaggregate(GP(obs_noise=0.1), y, t1, t2; output_period = Day(1))
        @test length(r_daily.signal) > length(r_monthly.signal)
        @test all(r_daily.std.data .>= 0.0)
        @test all(isfinite, r_daily.signal)
    end

    @testset "Posterior mean close to true signal" begin
        t1, t2 = make_monthly_intervals(Date(2020, 1, 1), 36)
        c = 7.5
        y = fill(c, 36)
        result = disaggregate(GP(obs_noise=1e-6), y, t1, t2)
        @test mean(result.signal) ≈ c atol=1.0
    end

    @testset "L1 loss returns valid result" begin
        t1, t2 = make_monthly_intervals(Date(2020, 1, 1), 24)
        y = [sin(2π * i / 12) for i in 1:24]
        result = disaggregate(GP(obs_noise=0.1), y, t1, t2; loss_norm = :L1)
        @test all(result.std.data .>= 0.0)
        @test all(isfinite, result.signal)
    end
end

# ─────────────────────────────────────────────────────────────────────────────
@testitem "All methods callable via disaggregate" setup=[TestHelpers] begin
    using TemporalDisaggregations
    using Dates
    using DimensionalData: DimStack, DimArray, Ti, hasdim, metadata

    t1, t2 = make_monthly_intervals(Date(2021, 1, 1), 12)
    y = ones(12)

    r_spline = disaggregate(Spline(), y, t1, t2)
    r_sin    = disaggregate(Sinusoid(), y, t1, t2)
    r_gp     = disaggregate(GP(), y, t1, t2)

    @test r_spline isa DimStack && hasdim(r_spline.signal, Ti)
    @test haskey(metadata(r_sin), :mean)
    @test r_gp.std isa DimArray
end

# ─────────────────────────────────────────────────────────────────────────────
@testitem "weights kwarg" setup=[TestHelpers] begin
    using TemporalDisaggregations
    using Dates
    using LinearAlgebra: norm
    using Statistics: cor

    @testset "Validation" begin
        t1, t2 = make_monthly_intervals(Date(2020, 1, 1), 6)
        y = ones(6)
        for method in [Spline(), Sinusoid(), GP()]
            @test_throws DimensionMismatch disaggregate(method, y, t1, t2; weights = ones(5))
            @test_throws ArgumentError    disaggregate(method, y, t1, t2; weights = [1,1,1, 0,1,1.0])
            @test_throws ArgumentError    disaggregate(method, y, t1, t2; weights = [1,1,1,-1,1,1.0])
        end
    end

    @testset "Uniform weights match no weights (all methods)" begin
        t1, t2 = make_monthly_intervals(Date(2020, 1, 1), 24)
        y = [sin(2π * i / 12) for i in 1:24]
        for method in [Spline(), Sinusoid(), GP(obs_noise=0.1)]
            r_none  = disaggregate(method, y, t1, t2)
            r_ones  = disaggregate(method, y, t1, t2; weights = ones(24))
            r_small = disaggregate(method, y, t1, t2; weights = fill(1e-4, 24))
            r_large = disaggregate(method, y, t1, t2; weights = fill(1e4,  24))
            @test r_none.signal.data  ≈ r_ones.signal.data  atol=1e-8
            @test r_none.std.data     ≈ r_ones.std.data     atol=1e-8
            @test r_small.signal.data ≈ r_ones.signal.data  atol=1e-6
            @test r_large.signal.data ≈ r_ones.signal.data  atol=1e-6
        end
    end

    @testset "Near-zero weight suppresses blunder" begin
        t1, t2 = make_monthly_intervals(Date(2020, 1, 1), 24)
        y_clean   = [sin(2π * i / 12) for i in 1:24]
        y_blunder = copy(y_clean)
        y_blunder[12] += 100.0
        r_truth      = disaggregate(Spline(), y_clean,   t1, t2)
        r_unweighted = disaggregate(Spline(), y_blunder, t1, t2)
        w = ones(24); w[12] = 1e-6
        r_weighted   = disaggregate(Spline(), y_blunder, t1, t2; weights = w)
        err_unw = norm(r_unweighted.signal.data - r_truth.signal.data)
        err_wei = norm(r_weighted.signal.data   - r_truth.signal.data)
        @test err_wei < err_unw
    end

    @testset "weights combined with L1 loss" begin
        t1, t2 = make_monthly_intervals(Date(2020, 1, 1), 24)
        y = [sin(2π * i / 12) for i in 1:24]
        w = rand(24) .+ 0.1
        r = disaggregate(Spline(), y, t1, t2; loss_norm = :L1, weights = w)
        @test all(isfinite, r.signal)
        @test all(r.std.data .>= 0)
    end

    @testset "GP L1 weights suppress blunder" begin
        t1, t2    = make_monthly_intervals(Date(2020, 1, 1), 24)
        y_clean   = [sin(2π * i / 12) for i in 1:24]
        y_blunder = copy(y_clean); y_blunder[12] += 100.0
        r_clean = disaggregate(GP(obs_noise=0.1), y_clean,   t1, t2; loss_norm=:L1)
        r_blow  = disaggregate(GP(obs_noise=0.1), y_blunder, t1, t2; loss_norm=:L1)
        w = ones(24); w[12] = 1e-6
        r_w     = disaggregate(GP(obs_noise=0.1), y_blunder, t1, t2; loss_norm=:L1, weights=w)
        @test norm(r_w.signal.data - r_clean.signal.data) <
              norm(r_blow.signal.data - r_clean.signal.data)
    end
end

# ─────────────────────────────────────────────────────────────────────────────
@testitem "Spline — analytical accuracy" setup=[TestHelpers] begin
    using TemporalDisaggregations
    using Dates
    using DimensionalData: Ti, dims, metadata
    using Statistics: mean, std
    using LinearAlgebra: norm

    @testset "Constant signal: near-exact recovery at low smoothness" begin
        c = 5.0
        t1, t2 = make_monthly_intervals(Date(2020, 1, 1), 24)
        y = fill(c, 24)
        r = disaggregate(Spline(smoothness=1e-8), y, t1, t2)
        @test mean(r.signal.data) ≈ c  atol=0.01
        @test all(abs.(r.signal.data .- c) .< 0.2)
        @test all(r.std.data .< 0.01)
    end

    @testset "Linear signal: near-exact recovery at low smoothness" begin
        t1, t2 = make_monthly_intervals(Date(2019, 1, 1), 36)
        t1_yr  = TD.yeardecimal.(t1)
        t2_yr  = TD.yeardecimal.(t2)
        a, b   = 3.0, 1.5
        t_mid  = (t1_yr[1] + t2_yr[end]) / 2
        y = [a + b * ((t1_yr[i] + t2_yr[i]) / 2 - t_mid) for i in eachindex(t1)]
        r = disaggregate(Spline(smoothness=1e-10), y, t1, t2)
        out_t  = TD.yeardecimal.(collect(dims(r.signal, Ti).val))
        x_true = a .+ b .* (out_t .- t_mid)
        @test maximum(abs.(r.signal.data .- x_true)) < 0.1
        @test all(r.std.data .< 0.05)
    end

    @testset "Round-trip: interval_average ≈ y (Day(1) output)" begin
        t1, t2 = make_monthly_intervals(Date(2020, 1, 1), 24)
        y = [2.0 + sin(2π * i / 12) for i in 1:24]
        r = disaggregate(Spline(smoothness=1e-6), y, t1, t2; output_period=Day(1))
        ŷ = interval_average(r, t1, t2)
        @test maximum(abs.(ŷ .- y)) < 0.05
    end

    @testset "Smoothness: larger smoothness → larger residual std" begin
        t1, t2 = make_monthly_intervals(Date(2020, 1, 1), 24)
        y = [2.0 + 1.5 * sin(2π * i / 12) for i in 1:24]
        r_low  = disaggregate(Spline(smoothness=1e-8), y, t1, t2)
        r_mid  = disaggregate(Spline(smoothness=1e0),  y, t1, t2)
        r_high = disaggregate(Spline(smoothness=1e4),  y, t1, t2)
        @test all(isfinite, r_low.signal.data)
        @test all(isfinite, r_mid.signal.data)
        @test all(isfinite, r_high.signal.data)
        @test mean(r_low.std.data) <= mean(r_mid.std.data)
        @test mean(r_mid.std.data) <= mean(r_high.std.data)
    end

    @testset "Tension: higher tension → lower second-difference roughness" begin
        t1, t2 = make_monthly_intervals(Date(2020, 1, 1), 24)
        y = [2.0 + sin(2π * i / 12) + 0.5 * sin(6π * i / 12) for i in 1:24]
        r0  = disaggregate(Spline(smoothness=1e-6, tension=0.0),  y, t1, t2)
        r5  = disaggregate(Spline(smoothness=1e-6, tension=5.0),  y, t1, t2)
        r50 = disaggregate(Spline(smoothness=1e-6, tension=50.0), y, t1, t2)
        @test all(isfinite, r5.signal.data)
        @test all(isfinite, r50.signal.data)
        rough(r) = sum(diff(diff(r.signal.data)).^2)
        @test rough(r50) <= rough(r0) + 0.5
    end

    @testset "penalty_order 1–4: all produce finite, non-negative-std results" begin
        t1, t2 = make_monthly_intervals(Date(2020, 1, 1), 24)
        y = [1.0 + 0.5 * sin(2π * i / 12) for i in 1:24]
        for po in [1, 2, 3, 4]
            r = disaggregate(Spline(smoothness=1e-4, penalty_order=po), y, t1, t2)
            @test all(isfinite, r.signal.data)
            @test all(r.std.data .>= 0)
        end
    end

    @testset "n_knots: nothing / 0 / 10 / 30 all valid; n_knots=3 throws" begin
        t1, t2 = make_monthly_intervals(Date(2020, 1, 1), 24)
        y = [sin(2π * i / 12) for i in 1:24]
        for nk in [nothing, 0, 10, 30]
            r = disaggregate(Spline(smoothness=1e-3, n_knots=nk), y, t1, t2)
            @test all(isfinite, r.signal.data)
            @test all(r.std.data .>= 0)
        end
        @test_throws ArgumentError disaggregate(Spline(n_knots=3), y, t1, t2)
    end

    @testset "L1 more robust than L2 for single large blunder" begin
        t1, t2    = make_monthly_intervals(Date(2020, 1, 1), 36)
        y_clean   = [sin(2π * i / 12) for i in 1:36]
        y_blunder = copy(y_clean); y_blunder[18] += 100.0
        r_truth = disaggregate(Spline(smoothness=1e-1), y_clean,   t1, t2)
        r_l2    = disaggregate(Spline(smoothness=1e-1), y_blunder, t1, t2; loss_norm=:L2)
        r_l1    = disaggregate(Spline(smoothness=1e-1), y_blunder, t1, t2; loss_norm=:L1)
        @test norm(r_l1.signal.data .- r_truth.signal.data) <
              norm(r_l2.signal.data .- r_truth.signal.data)
    end

    @testset "Heterogeneous weights (L2): explicit downweighting of blunder" begin
        t1, t2    = make_monthly_intervals(Date(2020, 1, 1), 24)
        y_clean   = [sin(2π * i / 12) for i in 1:24]
        y_blunder = copy(y_clean); y_blunder[12] += 100.0
        r_truth = disaggregate(Spline(smoothness=1e-6), y_clean,   t1, t2)
        r_unwt  = disaggregate(Spline(smoothness=1e-6), y_blunder, t1, t2)
        w = ones(24); w[12] = 1e-8
        r_wt    = disaggregate(Spline(smoothness=1e-6), y_blunder, t1, t2; weights=w)
        @test norm(r_wt.signal.data .- r_truth.signal.data) <
              norm(r_unwt.signal.data .- r_truth.signal.data)
    end

    @testset "Heterogeneous weights + L1: combined robustness" begin
        t1, t2    = make_monthly_intervals(Date(2020, 1, 1), 24)
        y_clean   = [2.0 + sin(2π * i / 12) for i in 1:24]
        y_blunder = copy(y_clean)
        y_blunder[6]  += 30.0; y_blunder[18] -= 30.0
        w = fill(1.0, 24); w[6] = 1e-6; w[18] = 1e-6
        r_truth = disaggregate(Spline(smoothness=1e-6), y_clean,   t1, t2)
        r_comb  = disaggregate(Spline(smoothness=1e-6), y_blunder, t1, t2;
                               loss_norm=:L1, weights=w)
        @test all(isfinite, r_comb.signal.data)
        @test norm(r_comb.signal.data .- r_truth.signal.data) < 1.0
    end

    @testset "std varies spatially with observation density" begin
        t1, t2 = make_monthly_intervals(Date(2020, 1, 1), 12)
        y = [sin(2π * i / 12) for i in 1:12]
        r = disaggregate(Spline(), y, t1, t2; output_period=Day(1))
        @test std(r.std.data) > 1e-8
        @test all(>=(0), r.std.data)
    end

    @testset "output_start / output_end clip the grid exactly" begin
        t1, t2 = make_monthly_intervals(Date(2019, 1, 1), 36)
        y      = ones(36)
        out_s  = Date(2019, 6, 1)
        out_e  = Date(2021, 6, 1)
        r = disaggregate(Spline(), y, t1, t2;
                         output_start=out_s, output_end=out_e)
        dates = collect(dims(r.signal, Ti).val)
        @test first(dates) == out_s
        @test last(dates)  <= out_e
        @test all(isfinite, r.signal.data)
        r_def = disaggregate(Spline(), y, t1, t2)
        @test first(collect(dims(r_def.signal, Ti).val)) == minimum(t1)
    end

    @testset "metadata reflects all constructor/kwarg arguments" begin
        t1, t2 = make_monthly_intervals(Date(2020, 1, 1), 12)
        r = disaggregate(
                Spline(smoothness=0.2, n_knots=20, penalty_order=2, tension=1.5),
                ones(12), t1, t2;
                loss_norm=:L1, output_period=Week(2))
        m = metadata(r)
        @test m[:method]        == :spline
        @test m[:smoothness]    == 0.2
        @test m[:n_knots]       == 20
        @test m[:penalty_order] == 2
        @test m[:tension]       == 1.5
        @test m[:loss_norm]     == :L1
        @test m[:output_period] == Week(2)
    end
end

# ─────────────────────────────────────────────────────────────────────────────
@testitem "Sinusoid — analytical accuracy" setup=[TestHelpers] begin
    using TemporalDisaggregations
    using Dates
    using DimensionalData: Ti, dims, metadata
    using Statistics: mean, std
    using LinearAlgebra: norm

    @testset "Constant signal: mean absorbed, trend and amplitude ≈ 0" begin
        μ = 5.0
        t1, t2 = make_monthly_intervals(Date(2020, 1, 1), 24)
        y = fill(μ, 24)
        r = disaggregate(Sinusoid(smoothness_interannual=1.0), y, t1, t2)
        @test metadata(r)[:mean]      ≈ μ   atol=0.05
        @test metadata(r)[:trend]     ≈ 0.0 atol=0.05
        @test metadata(r)[:amplitude] < 0.1
        @test all(abs.(r.signal.data .- μ) .< 0.2)
        @test all(r.std.data .< 0.05)
    end

    @testset "Linear trend: recovered from exact interval averages" begin
        t1, t2  = make_monthly_intervals(Date(2018, 1, 1), 36)
        t1_yr   = TD.yeardecimal.(t1)
        t2_yr   = TD.yeardecimal.(t2)
        t_ref   = (t1_yr[1] + t2_yr[end]) / 2
        β       = 2.0
        y = [β * ((t1_yr[i] + t2_yr[i]) / 2 - t_ref) for i in eachindex(t1)]
        r = disaggregate(Sinusoid(smoothness_interannual=1e-2), y, t1, t2)
        @test metadata(r)[:trend] ≈ β   atol=0.01
        @test metadata(r)[:mean]  ≈ 0.0 atol=0.05
    end

    @testset "Pure sinusoid: amplitude and phase recovered tightly" begin
        A, B       = 2.0, 3.0
        amp_true   = sqrt(A^2 + B^2)
        phase_true = mod(atan(B, A) / (2π), 1.0)
        t1, t2 = make_monthly_intervals(Date(2018, 1, 1), 48)
        t1_yr  = TD.yeardecimal.(t1)
        t2_yr  = TD.yeardecimal.(t2)
        y = [A * TD._interval_sin_integral(t1_yr[i], t2_yr[i]) +
             B * TD._interval_cos_integral(t1_yr[i], t2_yr[i])
             for i in eachindex(t1)]
        r = disaggregate(Sinusoid(smoothness_interannual=1e-8), y, t1, t2)
        @test metadata(r)[:amplitude] ≈ amp_true   atol=0.01
        @test metadata(r)[:phase]     ≈ phase_true atol=0.01
        out_t  = TD.yeardecimal.(collect(dims(r.signal, Ti).val))
        x_true = A .* sin.(2π .* out_t) .+ B .* cos.(2π .* out_t)
        @test maximum(abs.(r.signal.data .- x_true)) < 0.05
        @test all(r.std.data .< 0.01)
    end

    @testset "Combined signal: mean + trend + sinusoid all recovered" begin
        μ, β, A, B = 4.0, 0.5, 1.5, 2.0
        t1, t2  = make_monthly_intervals(Date(2018, 1, 1), 48)
        t1_yr   = TD.yeardecimal.(t1)
        t2_yr   = TD.yeardecimal.(t2)
        t_ref   = (t1_yr[1] + t2_yr[end]) / 2
        y = [μ + β * ((t1_yr[i] + t2_yr[i]) / 2 - t_ref) +
             A * TD._interval_sin_integral(t1_yr[i], t2_yr[i]) +
             B * TD._interval_cos_integral(t1_yr[i], t2_yr[i])
             for i in eachindex(t1)]
        r = disaggregate(Sinusoid(smoothness_interannual=1e-8), y, t1, t2)
        @test metadata(r)[:mean]      ≈ μ               atol=0.05
        @test metadata(r)[:trend]     ≈ β               atol=0.05
        @test metadata(r)[:amplitude] ≈ sqrt(A^2 + B^2) atol=0.05
    end

    @testset "smoothness_interannual: larger value suppresses anomalies" begin
        t1, t2 = make_monthly_intervals(Date(2020, 1, 1), 24)
        y = vcat(fill(6.0, 12), fill(4.0, 12))
        r_free  = disaggregate(Sinusoid(smoothness_interannual=1e-8), y, t1, t2)
        r_tight = disaggregate(Sinusoid(smoothness_interannual=1e3),  y, t1, t2)
        γ_free  = collect(values(metadata(r_free)[:interannual]))
        γ_tight = collect(values(metadata(r_tight)[:interannual]))
        @test sum(abs.(γ_free)) > sum(abs.(γ_tight))
    end

    @testset "Round-trip: interval_average ≈ y (Day(1) output)" begin
        t1, t2 = make_monthly_intervals(Date(2018, 1, 1), 48)
        t1_yr  = TD.yeardecimal.(t1)
        t2_yr  = TD.yeardecimal.(t2)
        A, B   = 1.5, 2.5
        y = [3.0 + A * TD._interval_sin_integral(t1_yr[i], t2_yr[i]) +
                   B * TD._interval_cos_integral(t1_yr[i], t2_yr[i])
             for i in eachindex(t1)]
        r = disaggregate(Sinusoid(smoothness_interannual=1e-8), y, t1, t2;
                         output_period=Day(1))
        ŷ = interval_average(r, t1, t2)
        @test maximum(abs.(ŷ .- y)) < 0.05
    end

    @testset "L1 more robust than L2 for single large blunder" begin
        t1, t2    = make_monthly_intervals(Date(2020, 1, 1), 36)
        y_clean   = [2.0 + sin(2π * i / 12) for i in 1:36]
        y_blunder = copy(y_clean); y_blunder[18] += 50.0
        r_truth = disaggregate(Sinusoid(smoothness_interannual=1e-6), y_clean,   t1, t2)
        r_l2    = disaggregate(Sinusoid(smoothness_interannual=1e-6), y_blunder, t1, t2; loss_norm=:L2)
        r_l1    = disaggregate(Sinusoid(smoothness_interannual=1e-6), y_blunder, t1, t2; loss_norm=:L1)
        @test norm(r_l1.signal.data .- r_truth.signal.data) <
              norm(r_l2.signal.data .- r_truth.signal.data)
    end

    @testset "Heterogeneous weights suppress blunder" begin
        t1, t2    = make_monthly_intervals(Date(2020, 1, 1), 24)
        y_clean   = [1.0 + 0.5 * sin(2π * i / 12) for i in 1:24]
        y_blunder = copy(y_clean); y_blunder[6] += 80.0
        r_truth = disaggregate(Sinusoid(smoothness_interannual=1e-2), y_clean,   t1, t2)
        r_unwt  = disaggregate(Sinusoid(smoothness_interannual=1e-2), y_blunder, t1, t2)
        w = ones(24); w[6] = 1e-8
        r_wt    = disaggregate(Sinusoid(smoothness_interannual=1e-2), y_blunder, t1, t2; weights=w)
        @test norm(r_wt.signal.data .- r_truth.signal.data) <
              norm(r_unwt.signal.data .- r_truth.signal.data)
    end

    @testset "std varies spatially with observation density" begin
        t1, t2 = make_monthly_intervals(Date(2020, 1, 1), 24)
        y = [1.0 + sin(2π * i / 12) for i in 1:24]
        r = disaggregate(Sinusoid(), y, t1, t2; output_period=Day(1))
        @test std(r.std.data) > 1e-8
        @test all(>=(0), r.std.data)
    end

    @testset "output_start / output_end clip the grid exactly" begin
        t1, t2 = make_monthly_intervals(Date(2019, 1, 1), 36)
        y      = ones(36)
        out_s  = Date(2019, 6, 1)
        out_e  = Date(2021, 6, 1)
        r = disaggregate(Sinusoid(), y, t1, t2;
                         output_start=out_s, output_end=out_e)
        dates = collect(dims(r.signal, Ti).val)
        @test first(dates) == out_s
        @test last(dates)  <= out_e
        @test all(isfinite, r.signal.data)
    end
end

# ─────────────────────────────────────────────────────────────────────────────
@testitem "GP — analytical accuracy" setup=[TestHelpers] begin
    using TemporalDisaggregations
    using Dates
    using DimensionalData: Ti, dims, metadata
    using Statistics: std
    using LinearAlgebra: norm
    using KernelFunctions: SqExponentialKernel, with_lengthscale

    @testset "Round-trip: interval_average ≈ y (Day(1) output, low noise)" begin
        t1, t2 = make_monthly_intervals(Date(2020, 1, 1), 24)
        y = [sin(2π * i / 12) for i in 1:24]
        r = disaggregate(GP(obs_noise=0.01), y, t1, t2; output_period=Day(1))
        ŷ = interval_average(r, t1, t2)
        @test maximum(abs.(ŷ .- y)) < 0.3
    end

    @testset "Posterior std decreases with lower obs_noise" begin
        t1, t2 = make_monthly_intervals(Date(2020, 1, 1), 24)
        y = fill(0.0, 24)
        r_low  = disaggregate(GP(obs_noise=0.01), y, t1, t2)
        r_high = disaggregate(GP(obs_noise=1.0),  y, t1, t2)
        @test all(>=(0), r_low.std.data)
        @test all(>=(0), r_high.std.data)
    end

    @testset "GP std varies spatially with observation density" begin
        t1, t2 = make_monthly_intervals(Date(2020, 1, 1), 12)
        y = [sin(2π * i / 12) for i in 1:12]
        r = disaggregate(GP(obs_noise=0.1), y, t1, t2; output_period=Day(1))
        @test std(r.std.data) > 1e-8
        @test all(>=(0), r.std.data)
    end

    @testset "n_quad consistency: n_quad=3 vs n_quad=15 for smooth signal" begin
        t1, t2 = make_monthly_intervals(Date(2020, 1, 1), 12)
        y = fill(0.0, 12)
        r3  = disaggregate(GP(obs_noise=0.1, n_quad=3),  y, t1, t2)
        r15 = disaggregate(GP(obs_noise=0.1, n_quad=15), y, t1, t2)
        @test maximum(abs.(r3.signal.data .- r15.signal.data)) < 0.01
        @test maximum(abs.(r3.std.data    .- r15.std.data))    < 0.01
    end

    @testset "Custom SqExponentialKernel produces finite non-negative results" begin
        t1, t2 = make_monthly_intervals(Date(2020, 1, 1), 24)
        y = [0.5 * sin(2π * i / 12) for i in 1:24]
        k_se = with_lengthscale(SqExponentialKernel(), 1 / 6)
        r = disaggregate(GP(kernel=k_se, obs_noise=0.1), y, t1, t2)
        @test all(isfinite, r.signal.data)
        @test all(r.std.data .>= 0)
    end

    @testset "L1 + explicit weights suppress blunder better than L2 alone" begin
        t1, t2    = make_monthly_intervals(Date(2020, 1, 1), 24)
        y_clean   = [sin(2π * i / 12) for i in 1:24]
        y_blunder = copy(y_clean); y_blunder[12] += 50.0
        r_clean = disaggregate(GP(obs_noise=0.1), y_clean,   t1, t2)
        r_l2    = disaggregate(GP(obs_noise=0.1), y_blunder, t1, t2; loss_norm=:L2)
        w = ones(24); w[12] = 1e-6
        r_l1w   = disaggregate(GP(obs_noise=0.1), y_blunder, t1, t2;
                               loss_norm=:L1, weights=w)
        @test norm(r_l1w.signal.data .- r_clean.signal.data) <
              norm(r_l2.signal.data  .- r_clean.signal.data)
    end

    @testset "Heterogeneous weights (L2) suppress blunder" begin
        t1, t2    = make_monthly_intervals(Date(2020, 1, 1), 24)
        y_clean   = [sin(2π * i / 12) for i in 1:24]
        y_blunder = copy(y_clean); y_blunder[6] += 60.0
        r_truth = disaggregate(GP(obs_noise=0.1), y_clean,   t1, t2)
        r_unwt  = disaggregate(GP(obs_noise=0.1), y_blunder, t1, t2)
        w = ones(24); w[6] = 1e-8
        r_wt    = disaggregate(GP(obs_noise=0.1), y_blunder, t1, t2; weights=w)
        @test norm(r_wt.signal.data .- r_truth.signal.data) <
              norm(r_unwt.signal.data .- r_truth.signal.data)
    end

    @testset "output_start / output_end clip the grid exactly" begin
        t1, t2 = make_monthly_intervals(Date(2019, 1, 1), 36)
        y      = [sin(2π * i / 12) for i in 1:36]
        out_s  = Date(2019, 6, 1)
        out_e  = Date(2021, 6, 1)
        r = disaggregate(GP(obs_noise=0.5), y, t1, t2;
                         output_start=out_s, output_end=out_e)
        dates = collect(dims(r.signal, Ti).val)
        @test first(dates) == out_s
        @test last(dates)  <= out_e
        @test all(isfinite, r.signal.data)
        @test all(r.std.data .>= 0)
    end

    @testset "metadata contains expected keys" begin
        t1, t2 = make_monthly_intervals(Date(2020, 1, 1), 12)
        y = ones(12)
        r = disaggregate(GP(obs_noise=0.5, n_quad=7), y, t1, t2; loss_norm=:L1)
        m = metadata(r)
        @test m[:method]    == :gp
        @test m[:obs_noise] == 0.5
        @test m[:n_quad]    == 7
        @test m[:loss_norm] == :L1
    end
end

# ─────────────────────────────────────────────────────────────────────────────
@testitem "GPKF method" setup=[TestHelpers] begin
    using TemporalDisaggregations
    using Dates
    using DimensionalData: DimStack, DimArray, Ti, hasdim, metadata
    using Statistics: cor, mean

    @testset "Returns DimStack with std ≥ 0" begin
        t1, t2 = make_monthly_intervals(Date(2020, 1, 1), 24)
        y = [sin(2π * i / 12) for i in 1:24]
        result = disaggregate(GPKF(obs_noise=0.1), y, t1, t2)
        @test result isa DimStack
        @test result.signal isa DimArray
        @test result.std    isa DimArray
        @test hasdim(result.signal, Ti)
        @test length(result.signal) == length(result.std)
        @test all(result.std.data .>= 0.0)
        @test all(isfinite, result.signal)
        @test all(isfinite, result.std)
    end

    @testset "Signal recovery on constant data" begin
        t1, t2 = make_monthly_intervals(Date(2020, 1, 1), 24)
        c = 5.0
        y = fill(c, 24)
        result = disaggregate(GPKF(obs_noise=1e-4), y, t1, t2)
        @test mean(result.signal) ≈ c atol=0.5
        @test all(isfinite, result.signal)
    end

    @testset "L1 loss returns valid result" begin
        t1, t2 = make_monthly_intervals(Date(2020, 1, 1), 24)
        y = [sin(2π * i / 12) for i in 1:24]
        result = disaggregate(GPKF(obs_noise=0.1), y, t1, t2; loss_norm = :L1)
        @test all(result.std.data .>= 0.0)
        @test all(isfinite, result.signal)
    end

    @testset "L1 vs L2 agree on clean data" begin
        t1, t2 = make_monthly_intervals(Date(2020, 1, 1), 24)
        y = [2.0 + sin(2π * i / 12) for i in 1:24]
        r_l2 = disaggregate(GPKF(obs_noise=0.1), y, t1, t2; loss_norm = :L2)
        r_l1 = disaggregate(GPKF(obs_noise=0.1), y, t1, t2; loss_norm = :L1)
        @test cor(r_l2.signal.data, r_l1.signal.data) > 0.95
    end

    @testset "output_period = Day(1) produces finer grid" begin
        t1, t2 = make_monthly_intervals(Date(2020, 1, 1), 12)
        y = ones(12)
        r_monthly = disaggregate(GPKF(obs_noise=0.1), y, t1, t2)
        r_daily   = disaggregate(GPKF(obs_noise=0.1), y, t1, t2; output_period = Day(1))
        @test length(r_daily.signal) > length(r_monthly.signal)
        @test all(isfinite, r_daily.signal)
    end

    @testset "Weighted observations: uniform weights match no weights" begin
        t1, t2 = make_monthly_intervals(Date(2020, 1, 1), 12)
        y = [sin(2π * i / 12) for i in 1:12]
        r_none = disaggregate(GPKF(obs_noise=0.1), y, t1, t2)
        r_ones = disaggregate(GPKF(obs_noise=0.1), y, t1, t2; weights = ones(12))
        @test r_none.signal.data ≈ r_ones.signal.data atol=1e-8
    end

    @testset "Input validation: DimensionMismatch and ArgumentError" begin
        t1, t2 = make_monthly_intervals(Date(2020, 1, 1), 6)
        y = ones(6)
        @test_throws DimensionMismatch disaggregate(GPKF(), y, t1, t2[1:5])
        @test_throws ArgumentError     disaggregate(GPKF(), y, t2, t1)
        @test_throws ArgumentError     disaggregate(GPKF(obs_noise=-1.0), y, t1, t2)
        @test_throws ArgumentError     disaggregate(GPKF(n_quad=2),       y, t1, t2)
        @test_throws ArgumentError     disaggregate(GPKF(), y, t1, t2; loss_norm=:L3)
        @test_throws DimensionMismatch disaggregate(GPKF(), y, t1, t2; weights=ones(5))
        @test_throws ArgumentError     disaggregate(GPKF(), y, t1, t2; weights=[1,1,1,0,1,1.0])
    end

    @testset "metadata contains expected keys" begin
        t1, t2 = make_monthly_intervals(Date(2020, 1, 1), 12)
        y = ones(12)
        r = disaggregate(GPKF(obs_noise=0.5, n_quad=7), y, t1, t2; loss_norm=:L1)
        m = metadata(r)
        @test m[:method]    == :gpkf
        @test m[:obs_noise] == 0.5
        @test m[:n_quad]    == 7
        @test m[:loss_norm] == :L1
    end
end

