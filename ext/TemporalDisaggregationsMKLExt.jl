module TemporalDisaggregationsMKLExt
# Loading MKL redirects OpenBLAS → Intel MKL (Intel CPUs, Linux/Windows).
# Users opt in by writing: using TemporalDisaggregations, MKL
using MKL
end
