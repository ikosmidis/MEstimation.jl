using Documenter
using GEEBRA

makedocs(
    sitename = "GEEBRA",
    authors = "Ioannis Kosmidis, Nicola Lunardon",
    format = Documenter.HTML(),
    modules = [GEEBRA],
    pages = [
        "Home" => "index.md",
        "Manual" => Any[
            "Guide" => "man/guide.md",
            "man/examples.md",
        ],
        "Library" => Any[
            "Public" => "lib/public.md",
        ],
        "contributing.md",
    ],
    doctest = false
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
