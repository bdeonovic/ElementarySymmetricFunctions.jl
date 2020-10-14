using ElementarySymmetricFunctions
using Documenter

makedocs(;
    modules=[ElementarySymmetricFunctions],
    authors="Benjamin Deonovic",
    repo="https://github.com/bdeonovic/ElementarySymmetricFunctions.jl/blob/{commit}{path}#L{line}",
    sitename="ElementarySymmetricFunctions.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://bdeonovic.github.io/ElementarySymmetricFunctions.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/bdeonovic/ElementarySymmetricFunctions.jl",
)
