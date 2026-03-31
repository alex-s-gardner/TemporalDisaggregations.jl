module TemporalDisaggregations

    using Dates
    using BasicBSpline
    using Statistics
    using LinearAlgebra
    using AbstractGPs
    using KernelFunctions
    using FastGaussQuadrature
    using DimensionalData: DimStack, DimArray, Ti, dims
    import DateFormats: yeardecimal
    using TemporalGPs: to_sde, SArrayStorage, ArrayStorage, ApproxPeriodicKernel

    include("utils.jl")
    include("methods.jl")
    include("disaggregate_spline.jl")
    include("disaggregate_sinusoid.jl")
    include("disaggregate_gp.jl")
    include("disaggregate_gpkf.jl")
    include("disaggregate.jl")
    include("precompile.jl")

    export disaggregate, yeardecimal,
        DisaggregationMethod, Spline, Sinusoid, GP, GPKF,
        interval_average,
        ApproxPeriodicKernel

end
