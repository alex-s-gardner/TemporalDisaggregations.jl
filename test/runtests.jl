using TemporalDisaggregations
using Test
using Dates
using DimensionalData: DimStack, DimArray, Ti, dims, hasdim, metadata
using LinearAlgebra
using Statistics

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
    @testset "Helper: _decimal_year" begin
        # Jan 1 of any year → exact integer
        @test TD.decimal_year(Date(2020, 1, 1)) == 2020.0
        @test TD.decimal_year(Date(2000, 1, 1)) == 2000.0

        # Mid-year for a non-leap year (2019): day 182/365 ≈ 0.4986
        # July 2 is day 183, so fraction = 182/365
        d_mid = Date(2019, 7, 2)   # day 183 → offset 182
        expected = 2019.0 + 182.0 / 365.0
        @test TD.decimal_year(d_mid) ≈ expected atol=1e-12

        # Leap year 2020: July 2 is day 184 → offset 183/366
        d_leap = Date(2020, 7, 2)
        expected_leap = 2020.0 + 183.0 / 366.0
        @test TD.decimal_year(d_leap) ≈ expected_leap atol=1e-12

        # DateTime variant: Jan 1 midnight
        @test TD.decimal_year(DateTime(2021, 1, 1, 0, 0, 0)) == 2021.0
    end

    # ─────────────────────────────────────────────────────────────────────────
    @testset "Helper: _monthly_decimal_year_grid" begin
        dates, times = TD._monthly_decimal_year_grid(2020.0, 2021.0)
        @test first(dates) == Date(2020, 1, 1)
        @test last(dates)  == Date(2021, 1, 1)
        @test length(dates) == 13
        @test times[1] ≈ 2020.0
        @test times[end] ≈ 2021.0
        @test issorted(times)
    end

    # ─────────────────────────────────────────────────────────────────────────
    @testset "Helper: _date_grid" begin
        dates, times = TD._date_grid(2020.0, 2021.0, Day(1))
        @test first(dates) == Date(2020, 1, 1)
        @test length(dates) > 300
        @test issorted(dates)
        @test all(isfinite, times)

        # Monthly step should match _monthly_decimal_year_grid result
        dates_m, times_m = TD._date_grid(2020.0, 2021.0, Month(1))
        dates_r, times_r = TD._monthly_decimal_year_grid(2020.0, 2021.0)
        @test dates_m == dates_r
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
        r = [1.0, 2.0, 0.0, -3.0]
        ε = 1e-6
        w = TD._irls_weights(r, ε)
        @test length(w) == 4
        @test all(w .> 0)
        @test w[1] ≈ 1.0 / (1.0 + ε)
        @test w[3] ≈ 1.0 / ε

        # Converged: identical vectors
        x = [1.0, 2.0, 3.0]
        @test TD._irls_converged(x, x)
        # Not converged: large change
        @test !TD._irls_converged(x .+ 1.0, x)
    end

    # ─────────────────────────────────────────────────────────────────────────
    @testset "Struct construction and defaults" begin
        s = Spline()
        @test s.smoothness == 1e-3
        @test s.n_knots === nothing
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
            @test all(Array(result.std) .>= 0)
        end

        @testset "L1 vs L2 agree on clean data" begin
            t1, t2 = make_monthly_intervals(Date(2020, 1, 1), 24)
            y = [3.0 + sin(2π * (i / 12.0)) for i in 1:24]
            r_l2 = disaggregate(Spline(), y, t1, t2; loss_norm = :L2)
            r_l1 = disaggregate(Spline(), y, t1, t2; loss_norm = :L1)
            @test cor(Array(r_l2.signal), Array(r_l1.signal)) > 0.99
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
            t1_dec = TD.decimal_year.(t1_dates)
            t2_dec = TD.decimal_year.(t2_dates)

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
            r_l2 = disaggregate(Sinusoid(), y, t1, t2; loss_norm = :L2)
            r_l1 = disaggregate(Sinusoid(), y, t1, t2; loss_norm = :L1)
            @test cor(Array(r_l2.signal), Array(r_l1.signal)) > 0.99
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
            @test all(Array(result.std) .>= 0)
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
            @test all(Array(result.std) .>= 0.0)
            @test all(isfinite, result.signal)
            @test all(isfinite, result.std)
        end

        @testset "output_period = Day(1) triggers kriging path" begin
            t1, t2 = make_monthly_intervals(Date(2020, 1, 1), 12)
            y = ones(12)
            r_monthly = disaggregate(GP(obs_noise=0.1), y, t1, t2)
            r_daily   = disaggregate(GP(obs_noise=0.1), y, t1, t2;
                                     output_period = Day(1))
            @test length(r_daily.signal) > length(r_monthly.signal)
            @test all(Array(r_daily.std) .>= 0.0)
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
                                  loss_norm = :L1)
            @test all(Array(result.std) .>= 0.0)
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

end
