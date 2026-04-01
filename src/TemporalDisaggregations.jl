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

    include("utils.jl")
    include("methods.jl")
    include("disaggregate_spline.jl")
    include("disaggregate_sinusoid.jl")
    include("disaggregate_gp.jl")
    include("disaggregate.jl")
    include("precompile.jl")

    export disaggregate, yeardecimal,
        DisaggregationMethod, Spline, Sinusoid, GP,
        interval_average

end
