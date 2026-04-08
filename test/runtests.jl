using TemporalDisaggregations
using Test
using Dates
using DimensionalData: DimStack, DimArray, Ti, dims, hasdim, metadata
using LinearAlgebra
using Statistics
using KernelFunctions: SqExponentialKernel, with_lengthscale
using LossFunctions: L1DistLoss, L2DistLoss, HuberLoss

const TD = TemporalDisaggregations

# ─────────────────────────────────────────────────────────────────────────────
# Helpers: synthetic data generators
# ─────────────────────────────────────────────────────────────────────────────

"""Build n non-overlapping monthly Date intervals starting from t_start."""
function make_monthly_intervals(t_start::Date, n::Int)
    t1 = [t_start + Month(i - 1) for i in 1:n]
    t2 = [t_start + Month(i)     for i in 1:n]
    return t1, t2
end

"""Exact average of f over [t1, t2] via 100-point quadrature (for reference)."""
function exact_average(f, t1::Float64, t2::Float64; npts=1000)
    ts = range(t1, t2; length=npts)
    return mean(f.(ts))
end

@testset "TemporalDisaggregations.jl" begin

    # ─────────────────────────────────────────────────────────────────────────
    @testset "Helper: _yeardecimal" begin
        # Jan 1 of any year → exact integer
        @test TD.yeardecimal(Date(2020, 1, 1)) == 2020.0
        @test TD.yeardecimal(Date(2000, 1, 1)) == 2000.0

        # Mid-year for a non-leap year (2019): day 182/365 ≈ 0.4986
        # July 2 is day 183, so fraction = 182/365
        d_mid = Date(2019, 7, 2)   # day 183 → offset 182
        expected = 2019.0 + 182.0 / 365.0
        @test TD.yeardecimal(d_mid) ≈ expected atol=1e-12

        # Leap year 2020: July 2 is day 184 → offset 183/366
        d_leap = Date(2020, 7, 2)
        expected_leap = 2020.0 + 183.0 / 366.0
        @test TD.yeardecimal(d_leap) ≈ expected_leap atol=1e-12

        # DateTime variant: Jan 1 midnight
        @test TD.yeardecimal(DateTime(2021, 1, 1, 0, 0, 0)) == 2021.0
    end

    # ─────────────────────────────────────────────────────────────────────────
    @testset "Helper: _date_grid" begin
        dates, times = TD._date_grid(Date(2020, 1, 1), Date(2021, 1, 1), Day(1))
        @test first(dates) == Date(2020, 1, 1)
        @test length(dates) > 300
        @test issorted(dates)
        @test all(isfinite, times)

        # Monthly step should produce 1st-of-month dates
        dates_m, _ = TD._date_grid(Date(2020, 1, 1), Date(2021, 1, 1), Month(1))
        @test first(dates_m) == Date(2020, 1, 1)
        @test last(dates_m)  == Date(2021, 1, 1)
        @test length(dates_m) == 13

        # start anchors the grid exactly
        dates_os, _ = TD._date_grid(Date(2020, 5, 15), Date(2021, 1, 1), Month(1))
        @test first(dates_os) == Date(2020, 5, 15)
        @test all(d -> day(d) == 15, dates_os)

        dates_w, _ = TD._date_grid(Date(2020, 3, 1), Date(2021, 1, 1), Week(2))
        @test first(dates_w) == Date(2020, 3, 1)
    end

    # ─────────────────────────────────────────────────────────────────────────
    @testset "Helper: _half_period" begin
        @test TD._half_period(Year(1))   == Month(6)
        @test TD._half_period(Year(2))   == Month(12)
        @test TD._half_period(Month(6))  == Month(3)
        @test TD._half_period(Month(2))  == Month(1)
        @test TD._half_period(Month(1))  == Week(2)
        @test TD._half_period(Week(4))   == Day(14)
        @test TD._half_period(Week(1))   == Day(4)
        @test TD._half_period(Day(4))    == Day(2)
        @test TD._half_period(Day(1))    == Day(1)   # floor
        @test TD._half_period(Hour(1))   == Day(1)   # sub-daily floor
    end

    # ─────────────────────────────────────────────────────────────────────────
    @testset "Helper: _difference_matrix" begin
        # Shape
        D1 = TD._difference_matrix(5, 1)
        @test size(D1) == (4, 5)

        # First-difference property: D1 * ones = zeros
        @test D1 * ones(5) ≈ zeros(4) atol=1e-14

        # First-difference entries: -1, 1 pattern
        @test D1[1, 1] ≈ -1.0
        @test D1[1, 2] ≈  1.0
        @test D1[2, 2] ≈ -1.0
        @test D1[2, 3] ≈  1.0

        # Second-difference: 1, -2, 1 pattern
        D2 = TD._difference_matrix(6, 2)
        @test size(D2) == (4, 6)
        @test D2[1, 1] ≈  1.0
        @test D2[1, 2] ≈ -2.0
        @test D2[1, 3] ≈  1.0

        # Third-difference: -1, 3, -3, 1 pattern
        D3 = TD._difference_matrix(7, 3)
        @test D3[1, 1] ≈ -1.0
        @test D3[1, 2] ≈  3.0
        @test D3[1, 3] ≈ -3.0
        @test D3[1, 4] ≈  1.0

        # Errors
        @test_throws ArgumentError TD._difference_matrix(5, 0)
        @test_throws ArgumentError TD._difference_matrix(5, 5)
    end

    # ─────────────────────────────────────────────────────────────────────────
    @testset "Helper: _irls_weights and _irls_converged" begin
        using LossFunctions
        r = [1.0, 2.0, 0.0, -3.0]
        ε = 1e-6
        loss = L1DistLoss()
        w = TD._irls_weights(r, loss, ε)
        @test length(w) == 4
        @test all(w .> 0)
        # For L1DistLoss, deriv = sign(r), so w = 1 / (|sign(r)| + ε) = 1 / (1 + ε)
        @test w[1] ≈ 1.0 / (1.0 + ε)
        @test w[3] ≈ 1.0 / (0.0 + ε)  # deriv(L1DistLoss(), 0.0) = 0.0

        # Converged: identical vectors
        x = [1.0, 2.0, 3.0]
        @test TD._irls_converged(x, x)
        # Not converged: large change
        @test !TD._irls_converged(x .+ 1.0, x)
    end

    # ─────────────────────────────────────────────────────────────────────────
    @testset "Struct construction and defaults" begin
        s = Spline()
        @test s.smoothness == 1e-1
        @test s.penalty_order == 3
        @test s.tension == 0.0

        sin_m = Sinusoid()
        @test sin_m.smoothness_interannual == 1e-2

        gp_m = GP()
        @test gp_m.obs_noise == 1.0
        @test gp_m.n_quad == 5

        # Type hierarchy
        @test Spline() isa DisaggregationMethod
        @test Sinusoid() isa DisaggregationMethod
        @test GP() isa DisaggregationMethod

        # Wrong kwargs caught at struct construction
        @test_throws MethodError Spline(obs_noise=1.0)
        @test_throws MethodError GP(smoothness=1e-3)
    end

    # ─────────────────────────────────────────────────────────────────────────
    @testset "Input validation" begin
        t1 = [Date(2020, 1, 1), Date(2020, 2, 1)]
        t2 = [Date(2020, 2, 1), Date(2020, 3, 1)]
        y  = [1.0, 2.0]

        # Length mismatch
        @test_throws DimensionMismatch disaggregate(Spline(), [1.0], t1, t2)
        @test_throws DimensionMismatch disaggregate(Spline(), y, [Date(2020,1,1)], t2)
        @test_throws DimensionMismatch disaggregate(Spline(), y, t1, [Date(2020,2,1)])

        # interval_end ≤ interval_start
        t2_bad = [Date(2020, 1, 1), Date(2020, 3, 1)]  # first end == start
        @test_throws ArgumentError disaggregate(Spline(), y, t1, t2_bad)
        t2_rev = [Date(2019, 12, 1), Date(2020, 3, 1)]  # first end < start
        @test_throws ArgumentError disaggregate(Spline(), y, t1, t2_rev)

        # Spline-specific: smoothness < 0
        @test_throws ArgumentError disaggregate(Spline(smoothness=-1.0), y, t1, t2)

        # GP-specific: obs_noise < 0
        @test_throws ArgumentError disaggregate(GP(obs_noise=-1.0), y, t1, t2)
    end

    # ─────────────────────────────────────────────────────────────────────────
    @testset "Spline method" begin

        @testset "Constant signal recovery" begin
            # 24 non-overlapping monthly intervals, constant value = 5.0
            t1, t2 = make_monthly_intervals(Date(2020, 1, 1), 24)
            y = fill(5.0, 24)
            result = disaggregate(Spline(smoothness=1e-8), y, t1, t2)
            @test result isa DimStack
            @test result.signal isa DimArray
            @test result.std    isa DimArray
            @test hasdim(result.signal, Ti)
            @test length(result.signal) == length(result.std)
            # Mean of recovered signal should be close to 5.0
            @test mean(result.signal) ≈ 5.0 atol=0.1
            # std should be non-negative
            @test all(result.std.data .>= 0)
        end

        @testset "L1 vs L2 agree on clean data" begin
            t1, t2 = make_monthly_intervals(Date(2020, 1, 1), 24)
            y = [3.0 + sin(2π * (i / 12.0)) for i in 1:24]
            r_l2 = disaggregate(Spline(), y, t1, t2; loss_norm = L2DistLoss())
            r_l1 = disaggregate(Spline(), y, t1, t2; loss_norm = L1DistLoss())
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
            # High tension should reduce peak-to-peak range toward piecewise-linear behaviour
            @test maximum(r_tension.signal) - minimum(r_tension.signal) <=
                  maximum(r_notension.signal) - minimum(r_notension.signal) + 0.5
        end
    end

    # ─────────────────────────────────────────────────────────────────────────
    @testset "Sinusoid method" begin

        @testset "Recover known amplitude and phase" begin
            # Pure sinusoid: x(t) = A*sin(2πt) + B*cos(2πt), no trend/interannual
            A_true, B_true = 3.0, 4.0
            amp_true   = sqrt(A_true^2 + B_true^2)   # 5.0
            phase_true = mod(atan(B_true, A_true) / (2π), 1.0)

            # Compute exact interval averages for 36 monthly intervals
            t1_dates, t2_dates = make_monthly_intervals(Date(2018, 1, 1), 36)
            t1_dec = TD.yeardecimal.(t1_dates)
            t2_dec = TD.yeardecimal.(t2_dates)

            # Exact average of A*sin(2πt)+B*cos(2πt) over [t1,t2]
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
            # Should contain years 2019, 2020, 2021
            for yr in [2019, 2020, 2021]
                @test haskey(metadata(result)[:interannual], yr)
            end
        end

        @testset "L1 vs L2 agree on clean data" begin
            t1, t2 = make_monthly_intervals(Date(2020, 1, 1), 24)
            y = [2.0 + sin(2π * i / 12) for i in 1:24]
            r_l2 = disaggregate(Sinusoid(), y, t1, t2; loss_norm = L2DistLoss())
            r_l1 = disaggregate(Sinusoid(), y, t1, t2; loss_norm = L1DistLoss())
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

    # ─────────────────────────────────────────────────────────────────────────
    @testset "GP method" begin

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
            r_daily   = disaggregate(GP(obs_noise=0.1), y, t1, t2;
                                     output_period = Day(1))
            @test length(r_daily.signal) > length(r_monthly.signal)
            @test all(r_daily.std.data .>= 0.0)
            @test all(isfinite, r_daily.signal)
        end

        @testset "Posterior mean close to true signal" begin
            # Pure constant signal: GP should recover it closely
            t1, t2 = make_monthly_intervals(Date(2020, 1, 1), 36)
            c = 7.5
            y = fill(c, 36)
            result = disaggregate(GP(obs_noise=1e-6), y, t1, t2)
            @test mean(result.signal) ≈ c atol=1.0
        end

        @testset "L1 loss returns valid result" begin
            t1, t2 = make_monthly_intervals(Date(2020, 1, 1), 24)
            y = [sin(2π * i / 12) for i in 1:24]
            result = disaggregate(GP(obs_noise=0.1), y, t1, t2;
                                  loss_norm = L1DistLoss())
            @test all(result.std.data .>= 0.0)
            @test all(isfinite, result.signal)
        end
    end

    # ─────────────────────────────────────────────────────────────────────────
    @testset "All methods callable via disaggregate" begin
        t1, t2 = make_monthly_intervals(Date(2021, 1, 1), 12)
        y = ones(12)

        r_spline = disaggregate(Spline(), y, t1, t2)
        r_sin    = disaggregate(Sinusoid(), y, t1, t2)
        r_gp     = disaggregate(GP(), y, t1, t2)

        @test r_spline isa DimStack && hasdim(r_spline.signal, Ti)
        @test haskey(metadata(r_sin), :mean)
        @test r_gp.std isa DimArray
    end

    # ─────────────────────────────────────────────────────────────────────────
    @testset "weights kwarg" begin

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
            w = rand(24) .+ 0.1        # random positive weights
            r = disaggregate(Spline(), y, t1, t2; loss_norm = L1DistLoss(), weights = w)
            @test all(isfinite, r.signal)
            @test all(r.std.data .>= 0)
        end

        @testset "GP L1 weights suppress blunder" begin
            t1, t2    = make_monthly_intervals(Date(2020, 1, 1), 24)
            y_clean   = [sin(2π * i / 12) for i in 1:24]
            y_blunder = copy(y_clean); y_blunder[12] += 100.0
            r_clean = disaggregate(GP(obs_noise=0.1), y_clean,   t1, t2; loss_norm=L1DistLoss())
            r_blow  = disaggregate(GP(obs_noise=0.1), y_blunder, t1, t2; loss_norm=L1DistLoss())
            w = ones(24); w[12] = 1e-6
            r_w     = disaggregate(GP(obs_noise=0.1), y_blunder, t1, t2; loss_norm=L1DistLoss(), weights=w)
            @test norm(r_w.signal.data - r_clean.signal.data) <
                  norm(r_blow.signal.data - r_clean.signal.data)
        end

    end

    # ─────────────────────────────────────────────────────────────────────────
    # ANALYTICAL TESTS: first-principles ground truth with tight tolerances
    # ─────────────────────────────────────────────────────────────────────────

    @testset "Spline — analytical accuracy" begin

        # ── Exact signal recovery ────────────────────────────────────────────

        @testset "Constant signal: near-exact recovery at low smoothness" begin
            # x(t) = c everywhere → every interval average equals c.
            # A quartic B-spline derivative can represent a constant exactly.
            # With λ → 0, the residuals collapse to zero.
            c = 5.0
            t1, t2 = make_monthly_intervals(Date(2020, 1, 1), 24)
            y = fill(c, 24)
            r = disaggregate(Spline(smoothness=1e-8), y, t1, t2)
            # Mean of output should be c; individual values may deviate slightly
            # at interval boundaries due to B-spline basis effects.
            @test mean(r.signal.data) ≈ c  atol=0.01
            @test all(abs.(r.signal.data .- c) .< 0.2)
            @test all(r.std.data .< 0.01)  # residual RMS ≈ 0 for perfect fit
        end

        @testset "Linear signal: near-exact recovery at low smoothness" begin
            # x(t) = a + b·(t − t_mid).  Interval average = a + b·(midpoint − t_mid).
            # A cubic B-spline (derivative of quartic) can represent any linear function
            # exactly, so the fit should be near-machine-precision with λ ≈ 0.
            t1, t2 = make_monthly_intervals(Date(2019, 1, 1), 36)
            t1_yr  = TD.yeardecimal.(t1)
            t2_yr  = TD.yeardecimal.(t2)
            a, b   = 3.0, 1.5                   # intercept + 1.5 units yr⁻¹
            t_mid  = (t1_yr[1] + t2_yr[end]) / 2
            y = [a + b * ((t1_yr[i] + t2_yr[i]) / 2 - t_mid) for i in eachindex(t1)]
            r = disaggregate(Spline(smoothness=1e-10), y, t1, t2)
            out_t  = TD.yeardecimal.(collect(dims(r.signal, Ti).val))
            x_true = a .+ b .* (out_t .- t_mid)
            @test maximum(abs.(r.signal.data .- x_true)) < 0.1
            @test all(r.std.data .< 0.05)
        end

        # ── Round-trip consistency ───────────────────────────────────────────

        @testset "Round-trip: interval_average ≈ y (Day(1) output)" begin
            # For smooth data with low smoothness and a daily output grid the
            # trapezoidal re-integration error is O(Δt²) ≪ 0.05.
            t1, t2 = make_monthly_intervals(Date(2020, 1, 1), 24)
            y = [2.0 + sin(2π * i / 12) for i in 1:24]
            r = disaggregate(Spline(smoothness=1e-6), y, t1, t2; output_period=Day(1))
            ŷ = interval_average(r, t1, t2)
            @test maximum(abs.(ŷ .- y)) < 0.05
        end

        # ── Parameter effects ────────────────────────────────────────────────

        @testset "Smoothness: larger smoothness → larger residual std" begin
            # Lower smoothness allows a better fit to data (smaller residual RMS).
            # Higher smoothness forces stronger regularization, trading fit quality
            # for smoothness. The residual std must increase monotonically with λ.
            t1, t2 = make_monthly_intervals(Date(2020, 1, 1), 24)
            y = [2.0 + 1.5 * sin(2π * i / 12) for i in 1:24]
            r_low  = disaggregate(Spline(smoothness=1e-8), y, t1, t2)
            r_mid  = disaggregate(Spline(smoothness=1e0),  y, t1, t2)
            r_high = disaggregate(Spline(smoothness=1e4),  y, t1, t2)
            # All outputs must be finite
            @test all(isfinite, r_low.signal.data)
            @test all(isfinite, r_mid.signal.data)
            @test all(isfinite, r_high.signal.data)
            # Higher smoothness → larger residuals → higher mean std
            @test mean(r_low.std.data) <= mean(r_mid.std.data)
            @test mean(r_mid.std.data) <= mean(r_high.std.data)
        end

        @testset "Tension: higher tension → lower second-difference roughness" begin
            # Tension adds a ‖D₂ a‖² penalty that suppresses curvature in x′(t).
            # The sum-of-squared second differences is a discrete proxy for ∫(x′′)² dt.
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


        # ── Robustness to outliers ───────────────────────────────────────────

        @testset "L1 more robust than L2 for single large blunder" begin
            # A 100-unit outlier at observation 18 of 36 with moderate smoothness.
            # With smoothness=1e-1, the B-spline cannot overfit the outlier. L1
            # IRLS down-weights the residual outlier; L2 does not, so L2 is pulled
            # toward the blunder more strongly.
            t1, t2    = make_monthly_intervals(Date(2020, 1, 1), 36)
            y_clean   = [sin(2π * i / 12) for i in 1:36]
            y_blunder = copy(y_clean); y_blunder[18] += 100.0
            r_truth = disaggregate(Spline(smoothness=1e-1), y_clean,   t1, t2)
            r_l2    = disaggregate(Spline(smoothness=1e-1), y_blunder, t1, t2; loss_norm=L2DistLoss())
            r_l1    = disaggregate(Spline(smoothness=1e-1), y_blunder, t1, t2; loss_norm=L1DistLoss())
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
            # Two blunders; both down-weighted and L1 loss active simultaneously.
            t1, t2    = make_monthly_intervals(Date(2020, 1, 1), 24)
            y_clean   = [2.0 + sin(2π * i / 12) for i in 1:24]
            y_blunder = copy(y_clean)
            y_blunder[6]  += 30.0; y_blunder[18] -= 30.0
            w = fill(1.0, 24); w[6] = 1e-6; w[18] = 1e-6
            r_truth = disaggregate(Spline(smoothness=1e-6), y_clean,   t1, t2)
            r_comb  = disaggregate(Spline(smoothness=1e-6), y_blunder, t1, t2;
                                   loss_norm=L1DistLoss(), weights=w)
            @test all(isfinite, r_comb.signal.data)
            @test norm(r_comb.signal.data .- r_truth.signal.data) < 1.0
        end

        # ── Output structure ─────────────────────────────────────────────────

        @testset "std varies spatially with observation density" begin
            # Sandwich variance: std should be higher in gaps and lower where coverage is dense.
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
            # Without clipping the grid starts at minimum(interval_start)
            r_def = disaggregate(Spline(), y, t1, t2)
            @test first(collect(dims(r_def.signal, Ti).val)) == minimum(t1)
        end

        @testset "metadata reflects all constructor/kwarg arguments" begin
            t1, t2 = make_monthly_intervals(Date(2020, 1, 1), 12)
            r = disaggregate(
                    Spline(smoothness=0.2, penalty_order=2, tension=1.5),
                    ones(12), t1, t2;
                    loss_norm=L1DistLoss(), output_period=Week(2))
            m = metadata(r)
            @test m[:method]        == :spline
            @test m[:smoothness]    == 0.2
            @test m[:penalty_order] == 2
            @test m[:tension]       == 1.5
            @test m[:loss_norm]     == "L1DistLoss"
            @test m[:output_period] == Week(2)
        end

    end  # Spline analytical

    # ─────────────────────────────────────────────────────────────────────────
    @testset "Sinusoid — analytical accuracy" begin

        # ── Exact parameter recovery ─────────────────────────────────────────

        @testset "Constant signal: mean absorbed, trend and amplitude ≈ 0" begin
            # x(t) = μ. All interval averages = μ.
            # Regularisation on γ pushes interannual anomalies → 0; μ absorbs the level.
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
            # x(t) = β·(t − t_ref).  The design-matrix trend column equals the
            # interval midpoint minus t_ref, so β is identified exactly.
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
            # x(t) = A·sin(2πt) + B·cos(2πt).  Interval averages are computed
            # from the closed-form _interval_sin/cos_integral helpers (same formulas
            # the model uses), so the system is well-specified.
            # With 48 monthly obs (6 params, ~42 dof), recovery should be near-exact.
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
            # Output signal should match A·sin(2πt) + B·cos(2πt)
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

        # ── smoothness_interannual effect ────────────────────────────────────

        @testset "smoothness_interannual: larger value suppresses anomalies" begin
            # Year 2020 averages = 6.0, year 2021 averages = 4.0 → true anomalies ≈ ±1.
            # Low smoothness: anomalies are free to fit; high smoothness: shrunk toward 0.
            # Total |γ| must be larger for the free case.
            t1, t2 = make_monthly_intervals(Date(2020, 1, 1), 24)
            y = vcat(fill(6.0, 12), fill(4.0, 12))
            r_free  = disaggregate(Sinusoid(smoothness_interannual=1e-8), y, t1, t2)
            r_tight = disaggregate(Sinusoid(smoothness_interannual=1e3),  y, t1, t2)
            γ_free  = collect(values(metadata(r_free)[:interannual]))
            γ_tight = collect(values(metadata(r_tight)[:interannual]))
            @test sum(abs.(γ_free)) > sum(abs.(γ_tight))
        end

        # ── Round-trip consistency ───────────────────────────────────────────

        @testset "Round-trip: interval_average ≈ y (Day(1) output)" begin
            # The Sinusoid model fits the data analytically (no quadrature).
            # With daily output, trapezoidal re-integration error is ≪ 0.05.
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

        # ── Robustness to outliers ───────────────────────────────────────────

        @testset "L1 more robust than L2 for single large blunder" begin
            t1, t2    = make_monthly_intervals(Date(2020, 1, 1), 36)
            y_clean   = [2.0 + sin(2π * i / 12) for i in 1:36]
            y_blunder = copy(y_clean); y_blunder[18] += 50.0
            r_truth = disaggregate(Sinusoid(smoothness_interannual=1e-6), y_clean,   t1, t2)
            r_l2    = disaggregate(Sinusoid(smoothness_interannual=1e-6), y_blunder, t1, t2; loss_norm=L2DistLoss())
            r_l1    = disaggregate(Sinusoid(smoothness_interannual=1e-6), y_blunder, t1, t2; loss_norm=L1DistLoss())
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

        # ── Output structure ─────────────────────────────────────────────────

        @testset "std varies spatially with observation density" begin
            # Sandwich variance: std should be higher in gaps and lower where coverage is dense.
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

    end  # Sinusoid analytical

    # ─────────────────────────────────────────────────────────────────────────
    @testset "GP — analytical accuracy" begin

        # ── Round-trip consistency ───────────────────────────────────────────

        @testset "Round-trip: interval_average ≈ y (Day(1) output, low noise)" begin
            # With obs_noise ≪ signal variance and a daily output grid, the GP
            # posterior mean tracks the observations; trapezoidal re-integration
            # should agree with y to within 0.3 (generous for GP approximation).
            t1, t2 = make_monthly_intervals(Date(2020, 1, 1), 24)
            y = [sin(2π * i / 12) for i in 1:24]
            r = disaggregate(GP(obs_noise=0.01), y, t1, t2; output_period=Day(1))
            ŷ = interval_average(r, t1, t2)
            @test maximum(abs.(ŷ .- y)) < 0.3
        end

        # ── Posterior uncertainty ────────────────────────────────────────────

        @testset "Posterior std decreases with lower obs_noise" begin
            # For fixed data and kernel, smaller obs_noise tightens the posterior.
            # Residual std should be finite and non-negative for all obs_noise values.
            t1, t2 = make_monthly_intervals(Date(2020, 1, 1), 24)
            y = fill(0.0, 24)
            r_low  = disaggregate(GP(obs_noise=0.01), y, t1, t2)
            r_high = disaggregate(GP(obs_noise=1.0),  y, t1, t2)
            @test all(>=(0), r_low.std.data)
            @test all(>=(0), r_high.std.data)
        end

        @testset "GP std varies spatially with observation density" begin
            # Sandwich variance: std should vary across the output grid.
            t1, t2 = make_monthly_intervals(Date(2020, 1, 1), 12)
            y = [sin(2π * i / 12) for i in 1:12]
            r = disaggregate(GP(obs_noise=0.1), y, t1, t2; output_period=Day(1))
            @test std(r.std.data) > 1e-8
            @test all(>=(0), r.std.data)
        end

        # ── Quadrature accuracy ──────────────────────────────────────────────

        @testset "n_quad consistency: n_quad=3 vs n_quad=15 for smooth signal" begin
            # Gauss-Legendre error is O((Δt/2)^{2n_quad}) for analytic integrands.
            # For n_quad=3 over a 1-month interval the error is already < 1e-6 for
            # smooth kernels. Results for n_quad=3 and n_quad=15 must agree to 0.01.
            t1, t2 = make_monthly_intervals(Date(2020, 1, 1), 12)
            y = fill(0.0, 12)
            r3  = disaggregate(GP(obs_noise=0.1, n_quad=3),  y, t1, t2)
            r15 = disaggregate(GP(obs_noise=0.1, n_quad=15), y, t1, t2)
            @test maximum(abs.(r3.signal.data .- r15.signal.data)) < 0.01
            @test maximum(abs.(r3.std.data    .- r15.std.data))    < 0.01
        end

        # ── Custom kernel ────────────────────────────────────────────────────

        @testset "Custom SqExponentialKernel produces finite non-negative results" begin
            t1, t2 = make_monthly_intervals(Date(2020, 1, 1), 24)
            y = [0.5 * sin(2π * i / 12) for i in 1:24]
            k_se = with_lengthscale(SqExponentialKernel(), 1 / 6)
            r = disaggregate(GP(kernel=k_se, obs_noise=0.1), y, t1, t2)
            @test all(isfinite, r.signal.data)
            @test all(r.std.data .>= 0)
        end

        # ── Robustness to outliers ───────────────────────────────────────────

        @testset "L1 + explicit weights suppress blunder better than L2 alone" begin
            # GP IRLS begins from uniform weights (= L2 solution) and converges
            # immediately; pure L1 == L2 for GP.  The effective robustness mechanism
            # is combining L1 loss with an explicit near-zero weight on the blunder.
            t1, t2    = make_monthly_intervals(Date(2020, 1, 1), 24)
            y_clean   = [sin(2π * i / 12) for i in 1:24]
            y_blunder = copy(y_clean); y_blunder[12] += 50.0
            r_clean = disaggregate(GP(obs_noise=0.1), y_clean,   t1, t2)
            r_l2    = disaggregate(GP(obs_noise=0.1), y_blunder, t1, t2; loss_norm=L2DistLoss())
            w = ones(24); w[12] = 1e-6
            r_l1w   = disaggregate(GP(obs_noise=0.1), y_blunder, t1, t2;
                                   loss_norm=L1DistLoss(), weights=w)
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

        # ── Output structure ─────────────────────────────────────────────────

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
            r = disaggregate(GP(obs_noise=0.5, n_quad=7), y, t1, t2; loss_norm=L1DistLoss())
            m = metadata(r)
            @test m[:method]    == :gp
            @test m[:obs_noise] == 0.5
            @test m[:n_quad]    == 7
            @test m[:loss_norm] == "L1DistLoss"
        end

    end  # GP analytical

    # ─────────────────────────────────────────────────────────────────────────
    @testset "Huber Loss" begin

        @testset "Basic functionality" begin
            @testset "Spline Huber returns valid result" begin
                t1, t2 = make_monthly_intervals(Date(2020, 1, 1), 24)
                y = [3.0 + sin(2π * i / 12) for i in 1:24]

                r = disaggregate(Spline(), y, t1, t2; loss_norm = HuberLoss(1.345))
                @test all(isfinite, r.signal)
                @test all(r.std.data .>= 0.0)
                @test length(r.signal) > length(y)
                @test metadata(r)[:loss_norm] == "HuberLoss with \$\\alpha\$ = 1.345"
            end

            @testset "Sinusoid Huber returns valid result" begin
                t1, t2 = make_monthly_intervals(Date(2020, 1, 1), 24)
                y = [2.0 + sin(2π * i / 12) for i in 1:24]

                r = disaggregate(Sinusoid(), y, t1, t2; loss_norm = HuberLoss(1.345))
                @test all(isfinite, r.signal)
                @test all(r.std.data .>= 0.0)
                @test metadata(r)[:loss_norm] == "HuberLoss with \$\\alpha\$ = 1.345"
            end

            @testset "GP Huber returns valid result" begin
                t1, t2 = make_monthly_intervals(Date(2020, 1, 1), 24)
                y = [sin(2π * i / 12) for i in 1:24]

                r = disaggregate(GP(obs_noise=0.1), y, t1, t2; loss_norm = HuberLoss(1.345))
                @test all(isfinite, r.signal)
                @test all(r.std.data .>= 0.0)
                @test metadata(r)[:loss_norm] == "HuberLoss with \$\\alpha\$ = 1.345"
            end
        end

        @testset "Custom delta parameter" begin
            t1, t2 = make_monthly_intervals(Date(2020, 1, 1), 24)
            y = [sin(2π * i / 12) for i in 1:24]

            r1 = disaggregate(Spline(), y, t1, t2; loss_norm=HuberLoss(1.0))
            r2 = disaggregate(Spline(), y, t1, t2; loss_norm=HuberLoss(2.0))

            @test all(isfinite, r1.signal)
            @test all(isfinite, r2.signal)
            # Different deltas should give different results
            @test !all(r1.signal.data .≈ r2.signal.data)
        end

        @testset "Clean data behavior" begin
            # On clean data (no outliers), Huber should be similar to L2
            t1, t2 = make_monthly_intervals(Date(2020, 1, 1), 36)
            y_clean = [3.0 + sin(2π * i / 12) + 0.1 * randn() for i in 1:36]

            @testset "Spline: Huber ≈ L2 on clean data" begin
                r_l2    = disaggregate(Spline(smoothness=1e-1), y_clean, t1, t2; loss_norm=L2DistLoss())
                r_huber = disaggregate(Spline(smoothness=1e-1), y_clean, t1, t2; loss_norm=HuberLoss(1.345))

                # Should be highly correlated on clean data
                @test cor(r_l2.signal.data, r_huber.signal.data) > 0.99
            end

            @testset "Sinusoid: Huber ≈ L2 on clean data" begin
                r_l2    = disaggregate(Sinusoid(), y_clean, t1, t2; loss_norm=L2DistLoss())
                r_huber = disaggregate(Sinusoid(), y_clean, t1, t2; loss_norm=HuberLoss(1.345))

                @test cor(r_l2.signal.data, r_huber.signal.data) > 0.99
            end

            @testset "GP: Huber ≈ L2 on clean data" begin
                y_short = y_clean[1:24]
                t1_s, t2_s = t1[1:24], t2[1:24]

                r_l2    = disaggregate(GP(obs_noise=0.5), y_short, t1_s, t2_s; loss_norm=L2DistLoss())
                r_huber = disaggregate(GP(obs_noise=0.5), y_short, t1_s, t2_s; loss_norm=HuberLoss(1.345))

                @test cor(r_l2.signal.data, r_huber.signal.data) > 0.95
            end
        end

        @testset "Outlier robustness" begin
            # Huber should be more robust than L2, comparable to L1
            t1, t2 = make_monthly_intervals(Date(2020, 1, 1), 36)
            y_clean = [sin(2π * i / 12) for i in 1:36]
            y_outlier = copy(y_clean)
            y_outlier[18] += 50.0  # Large outlier in the middle

            @testset "Spline: Huber more robust than L2" begin
                m = Spline(smoothness=1e-1)
                r_clean = disaggregate(m, y_clean, t1, t2; loss_norm=L2DistLoss())
                r_l2    = disaggregate(m, y_outlier, t1, t2; loss_norm=L2DistLoss())
                r_huber = disaggregate(m, y_outlier, t1, t2; loss_norm=HuberLoss(1.345))

                # Huber should be closer to clean truth than L2
                err_l2    = norm(r_l2.signal.data - r_clean.signal.data)
                err_huber = norm(r_huber.signal.data - r_clean.signal.data)
                @test err_huber < err_l2
            end

            @testset "Sinusoid: Huber more robust than L2" begin
                m = Sinusoid(smoothness_interannual=1e-6)
                r_clean = disaggregate(m, y_clean, t1, t2; loss_norm=L2DistLoss())
                r_l2    = disaggregate(m, y_outlier, t1, t2; loss_norm=L2DistLoss())
                r_huber = disaggregate(m, y_outlier, t1, t2; loss_norm=HuberLoss(1.345))

                err_l2    = norm(r_l2.signal.data - r_clean.signal.data)
                err_huber = norm(r_huber.signal.data - r_clean.signal.data)
                @test err_huber < err_l2
            end

            @testset "GP: Huber more robust than L2" begin
                y_short_c = y_clean[1:24]
                y_short_o = y_outlier[1:24]
                t1_s, t2_s = t1[1:24], t2[1:24]

                m = GP(obs_noise=0.1)
                r_clean = disaggregate(m, y_short_c, t1_s, t2_s; loss_norm=L2DistLoss())
                r_l2    = disaggregate(m, y_short_o, t1_s, t2_s; loss_norm=L2DistLoss())
                r_huber = disaggregate(m, y_short_o, t1_s, t2_s; loss_norm=HuberLoss(1.345))

                err_l2    = norm(r_l2.signal.data - r_clean.signal.data)
                err_huber = norm(r_huber.signal.data - r_clean.signal.data)
                @test err_huber < err_l2
            end

            @testset "Huber vs L1 comparison" begin
                # Huber and L1 should give similar robustness
                m = Spline(smoothness=1e-1)
                r_clean = disaggregate(m, y_clean, t1, t2; loss_norm=L2DistLoss())
                r_l1    = disaggregate(m, y_outlier, t1, t2; loss_norm=L1DistLoss())
                r_huber = disaggregate(m, y_outlier, t1, t2; loss_norm=HuberLoss(1.345))

                # Both should be similarly close to truth (within 20% of each other)
                err_l1    = norm(r_l1.signal.data - r_clean.signal.data)
                err_huber = norm(r_huber.signal.data - r_clean.signal.data)
                @test abs(err_l1 - err_huber) / max(err_l1, err_huber) < 0.2
            end
        end

        @testset "Delta parameter effect" begin
            # Smaller delta = more aggressive downweighting = closer to L1
            # Larger delta = more observations treated as inliers = closer to L2
            t1, t2 = make_monthly_intervals(Date(2020, 1, 1), 36)
            y_clean = [sin(2π * i / 12) for i in 1:36]
            y_outlier = copy(y_clean)
            y_outlier[18] += 5.0  # Moderate outlier

            r_l2     = disaggregate(Spline(smoothness=1e-1), y_outlier, t1, t2; loss_norm=L2DistLoss())
            r_small  = disaggregate(Spline(smoothness=1e-1), y_outlier, t1, t2; loss_norm=HuberLoss(0.5))
            r_medium = disaggregate(Spline(smoothness=1e-1), y_outlier, t1, t2; loss_norm=HuberLoss(1.345))
            r_large  = disaggregate(Spline(smoothness=1e-1), y_outlier, t1, t2; loss_norm=HuberLoss(5.0))

            # Larger delta should be closer to L2
            cor_small  = cor(r_small.signal.data, r_l2.signal.data)
            cor_medium = cor(r_medium.signal.data, r_l2.signal.data)
            cor_large  = cor(r_large.signal.data, r_l2.signal.data)

            @test cor_small < cor_medium < cor_large
        end

        @testset "Heterogeneous weights" begin
            # Huber should work with per-observation weights
            t1, t2 = make_monthly_intervals(Date(2020, 1, 1), 24)
            y = [sin(2π * i / 12) for i in 1:24]
            w = rand(24) .+ 0.5  # Random positive weights

            r = disaggregate(Spline(), y, t1, t2; loss_norm=HuberLoss(1.345), weights=w)
            @test all(isfinite, r.signal)
            @test all(r.std.data .>= 0)
        end

        @testset "Edge cases" begin
            @testset "Constant data" begin
                t1, t2 = make_monthly_intervals(Date(2020, 1, 1), 12)
                y = ones(12)

                r = disaggregate(Spline(), y, t1, t2; loss_norm=HuberLoss(1.345))
                @test all(isfinite, r.signal)
                # Should approximate constant
                @test std(r.signal.data) < 0.1
            end

            @testset "Zero delta validation" begin
                # HuberLoss delta must be > 0 (enforced by LossFunctions.jl)
                @test_throws ErrorException HuberLoss(0.0)
                @test_throws ErrorException HuberLoss(-1.0)
            end

            @testset "Very large residuals" begin
                # Test numerical stability with extreme outliers
                t1, t2 = make_monthly_intervals(Date(2020, 1, 1), 24)
                y = [sin(2π * i / 12) for i in 1:24]
                y[12] = 1e6  # Extreme outlier

                r = disaggregate(Spline(), y, t1, t2; loss_norm=HuberLoss(1.345))
                @test all(isfinite, r.signal)
                @test all(r.std.data .>= 0)
            end
        end

        @testset "IRLS convergence" begin
            # Huber should converge (not hit 50-iteration limit on reasonable data)
            t1, t2 = make_monthly_intervals(Date(2020, 1, 1), 24)
            y = [sin(2π * i / 12) + 0.1 * randn() for i in 1:24]

            # Should converge without issues
            r = disaggregate(Spline(), y, t1, t2; loss_norm=HuberLoss(1.345))
            @test all(isfinite, r.signal)
            # No direct way to check iteration count, but should not error
        end

    end  # Huber Loss

    @testset "LossFunctions.jl integration" begin
        # Verify that LossFunctions.jl integration works correctly
        t1, t2 = make_monthly_intervals(Date(2020, 1, 1), 36)
        y = [sin(2π * i / 12) + 0.1 * randn() for i in 1:36]

        # Add a blunder for L1/Huber sensitivity
        y[10] += 5 * std(y)

        # Test all three methods with all three loss types
        @testset "All methods × all losses" begin
            for method in [Spline(), Sinusoid(), GP(obs_noise=0.1)]
                for (loss, expected_name) in [(L1DistLoss(), "L1DistLoss"), (L2DistLoss(), "L2DistLoss"), (HuberLoss(1.345), "HuberLoss with \$\\alpha\$ = 1.345")]
                    r = disaggregate(method, y, t1, t2; loss_norm=loss)

                    # Basic smoke test: result should have signal and std
                    @test haskey(r, :signal)
                    @test haskey(r, :std)
                    @test length(r.signal) > 0

                    # Metadata should record loss type
                    @test metadata(r)[:loss_norm] == expected_name
                end
            end
        end

        # Numerical stability: IRLS should converge
        @testset "IRLS convergence" begin
            for loss_norm in [L1DistLoss(), HuberLoss(1.345)]
                r = disaggregate(Spline(), y, t1, t2; loss_norm=loss_norm)
                @test !isnan(r.signal[1])
                @test all(isfinite.(r.signal))
                @test all(isfinite.(r.std))
            end
        end

        # Custom Huber delta parameter
        @testset "Custom Huber delta" begin
            r1 = disaggregate(Spline(), y, t1, t2; loss_norm=HuberLoss(0.5))
            r2 = disaggregate(Spline(), y, t1, t2; loss_norm=HuberLoss(2.0))

            # Different delta values should produce different results
            @test !isapprox(r1.signal[1], r2.signal[1])
        end
    end  # LossFunctions.jl integration

    @testset "IRLS tolerance validation" begin
        n = 50
        t1, t2 = make_monthly_intervals(Date(2020, 1, 1), n)
        y = randn(n)

        # Valid tolerances
        @test disaggregate(Spline(), y, t1, t2; irls_tol=1e-8) isa DimStack
        @test disaggregate(Spline(), y, t1, t2; irls_tol=1e-4) isa DimStack
        @test disaggregate(Spline(), y, t1, t2; irls_tol=1e-12) isa DimStack

        # Invalid tolerances
        @test_throws ArgumentError disaggregate(Spline(), y, t1, t2; irls_tol=0.0)
        @test_throws ArgumentError disaggregate(Spline(), y, t1, t2; irls_tol=-1e-8)

        # Test all three methods accept the parameter
        @test disaggregate(Sinusoid(), y, t1, t2; irls_tol=1e-6) isa DimStack
        @test disaggregate(GP(), y, t1, t2; irls_tol=1e-6) isa DimStack
    end  # IRLS tolerance validation

    @testset "IRLS max_iter validation" begin
        n = 50
        t1, t2 = make_monthly_intervals(Date(2020, 1, 1), n)
        y = randn(n)

        # Valid max_iter values
        @test disaggregate(Spline(), y, t1, t2; irls_max_iter=50) isa DimStack
        @test disaggregate(Spline(), y, t1, t2; irls_max_iter=10) isa DimStack
        @test disaggregate(Spline(), y, t1, t2; irls_max_iter=100) isa DimStack
        @test disaggregate(Spline(), y, t1, t2; irls_max_iter=1) isa DimStack

        # Invalid max_iter values
        @test_throws ArgumentError disaggregate(Spline(), y, t1, t2; irls_max_iter=0)
        @test_throws ArgumentError disaggregate(Spline(), y, t1, t2; irls_max_iter=-1)

        # Test all three methods accept the parameter
        @test disaggregate(Sinusoid(), y, t1, t2; irls_max_iter=20) isa DimStack
        @test disaggregate(GP(), y, t1, t2; irls_max_iter=20) isa DimStack

        # Test both parameters together
        @test disaggregate(Spline(), y, t1, t2; irls_tol=1e-6, irls_max_iter=20) isa DimStack
    end  # IRLS max_iter validation

end
