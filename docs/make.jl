using TemporalDisaggregations
using Documenter, DocumenterVitepress

makedocs(;
    modules = [TemporalDisaggregations],
    authors = "Alex Gardner",
    repo = "https://github.com/alex-s-gardner/TemporalDisaggregations.jl",
    sitename = "TemporalDisaggregations.jl",
    format = DocumenterVitepress.MarkdownVitepress(
        repo = "https://github.com/alex-s-gardner/TemporalDisaggregations.jl",
        devbranch = "main",
        devurl = "dev",
    ),
    pages = [
        "Home" => "index.md",
        "Getting Started" => "getting_started.md",
        "Methods" => "methods.md",
        "API Reference" => "api.md",
    ],
    warnonly = true,
)

DocumenterVitepress.deploydocs(;
    repo = "github.com/alex-s-gardner/TemporalDisaggregations.jl",
    push_preview = true,
)
