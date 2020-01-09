using Documenter
using GEEBRA

makedocs(
    sitename = "GEEBRA",
    authors = "Ioannis Kosmidis, Nicola Lunardon",
    format = Documenter.HTML(),
    modules = [GEEBRA],
    pages = [
        "Home" => "index.md",
        "Examples" => "man/examples.md",
        "Documentation" => Any[
            "Public" => "lib/public.md",
            "Internal" => "lib/internal.md",
        ]
    ],
    doctest = false
)

# deploydocs(
#     repo = "github.com/ikosmidis/GEEBRA.jl.git",
#     target = "build",
#     push_preview = true,
# )

