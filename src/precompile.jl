# Run a small workload during precompilation so the main call paths are
# compiled into the package cache.  ccall(:jl_generating_output,...) is true
# only while Julia is writing the .ji file, so this adds zero overhead at
# runtime.
if ccall(:jl_generating_output, Cint, ()) == 1
    let
        t1 = [Date(2020, m, 1)            for m in 1:12]
        t2 = [Date(2020, m, 1) + Month(1) for m in 1:12]
        y  = Float64.(1:12)
        w  = fill(1.0, 12)

        # Default L2 paths (Date boundaries, Month output)
        r_sp   = disaggregate(Spline(),   y, t1, t2; output_period = Month(1))
        r_sin  = disaggregate(Sinusoid(), y, t1, t2; output_period = Month(1))
        r_gp   = disaggregate(GP(),       y, t1, t2; output_period = Month(1))
        r_gpkf = disaggregate(GPKF(),     y, t1, t2; output_period = Month(1))

        # L1 loss path (IRLS loop — all methods share helpers)
        disaggregate(Spline(),   y, t1, t2; loss_norm = :L1)
        disaggregate(Sinusoid(), y, t1, t2; loss_norm = :L1)
        disaggregate(GP(),       y, t1, t2; loss_norm = :L1)
        disaggregate(GPKF(),     y, t1, t2; loss_norm = :L1)

        # Weighted observations path
        disaggregate(Spline(),   y, t1, t2; weights = w)
        disaggregate(Sinusoid(), y, t1, t2; weights = w)
        disaggregate(GP(),       y, t1, t2; weights = w)
        disaggregate(GPKF(),     y, t1, t2; weights = w)

        # interval_average (exported post-processing function)
        interval_average(r_sp,   t1, t2)
        interval_average(r_sin,  t1, t2)
        interval_average(r_gp,   t1, t2)
        interval_average(r_gpkf, t1, t2)

        # DateTime boundaries (separate _date_grid specialisation)
        dt1 = DateTime.(t1)
        dt2 = DateTime.(t2)
        disaggregate(Spline(),   y, dt1, dt2)
        disaggregate(Sinusoid(), y, dt1, dt2)
        disaggregate(GP(),       y, dt1, dt2)
        disaggregate(GPKF(),     y, dt1, dt2)
    end
end
