module TemporalDisaggregations

    using Dates
    using BasicBSpline
    using Statistics
    using LinearAlgebra
    using AbstractGPs
    using KernelFunctions
    using FastGaussQuadrature
    using DimensionalData: DimStack, DimArray, Ti
    import DateFormats: yeardecimal

    include("utils.jl")
    include("methods.jl")
    include("disaggregate_spline.jl")
    include("disaggregate_sinusoid.jl")
    include("disaggregate_gp.jl")
    include("disaggregate.jl")

    export disaggregate, yeardecimal,
        DisaggregationMethod, Spline, Sinusoid, GP

end
