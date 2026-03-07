module TemporalDisaggregations

    using DimensionalData: DimStack, DimArray, Ti

    include("utils.jl")
    include("methods.jl")
    include("disaggregate_spline.jl")
    include("disaggregate_sinusoid.jl")
    include("disaggregate_gp.jl")
    include("disaggregate.jl")

    export disaggregate, decimal_year,
        DisaggregationMethod, Spline, Sinusoid, GP

end
