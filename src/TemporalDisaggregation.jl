module TemporalDisaggregation

include("disaggregate_spline.jl")
include("disaggregate_sinusoid.jl")
include("disaggregate_gp.jl")
include("disaggregate.jl")

export disaggregate

end
