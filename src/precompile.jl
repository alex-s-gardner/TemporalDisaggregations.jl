using PrecompileTools

@compile_workload begin
    t1 = [Date(2020, m, 1)      for m in 1:12]
    t2 = [Date(2020, m, 1) + Month(1) for m in 1:12]
    y  = Float64.(1:12)

    disaggregate(Spline(),    y, t1, t2; output_period = Month(1))
    disaggregate(Sinusoid(),  y, t1, t2; output_period = Month(1))
    disaggregate(GP(),        y, t1, t2; output_period = Month(1))
end
