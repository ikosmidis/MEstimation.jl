# [GEEBRA.jl](https://github.com/ikosmidis/GEEBRA.jl)

## Authors

| [**Ioannis Kosmidis**](http://www.ikosmidis.com) | **(author, maintainer)** |
--- | ---
| [**Nicola Lunardon**](https://www.unimib.it/nicola-lunardon) | **(author)** |

## Licence

[MIT License](https://github.com/ikosmidis/GEEBRA.jl/blob/master/LICENSE.md)

## Package description

**GEEBRA** is a Julia package that implements ``M``-estimation for
statistical models, either by solving estimating equations or by
maximizing inference objectives, like
[likelihoods](https://en.wikipedia.org/wiki/Likelihood_function) and
composite likelihoods (see, [Varin et al,
2011](http://www3.stat.sinica.edu.tw/statistica/oldpdf/A21n11.pdf),
for a review), using user-specified templates of the estimating
function or the objective functions contributions.

A key feature is the use of only those templates and forward mode
automatic differentiation (as implemented in
[**ForwardDiff**](https://github.com/JuliaDiff/ForwardDiff.jl)) to
provide methods for **reduced-bias ``M``-estimation**
(**RB``M``-estimation**). RB``M``-estimation takes place either through the
adjustment of the estimating equations or the penalization of the
objectives, or the subtraction of an estimate of the bias of the
``M``-estimator from the ``M``-estimates.

See the
[examples](https://ikosmidis.github.io/GEEBRA.jl/dev/man/examples/)
for a showcase of the functionaly **GEEBRA** provides.

See
[NEWS.md](https://github.com/ikosmidis/GEEBRA.jl/blob/master/NEWS.md)
for changes, bug fixes and enhancements.

## **GEEBRA** templates

**GEEBRA** has been designed so that the only requirements from the user are to:
1. implement a [Julia composite type](https://docs.julialang.org/en/v1/manual/types/index.html) for
   the data;
2. implement a function for computing the number of observations from
   the data object;
3. implement a function for calculating the contribution to the
   estimating function or to the objective function from a single
   observation that has arguments the parameter vector, the data
   object, and the observation index;
4. specify a GEEBRA template (using
   [`estimating_function_template`](@ref) for estimating functions and
   [`objective_function_template`](@ref) for objective function) that
   has fields the functions for computing the contributions to the
   estimating functions or to the objective, and the number of
   observations.

**GEEBRA**, then, can estimate the unknown parameters by either
``M``-estimation or RB``M``-estimation.

## Examples

```@contents
Pages = [
    "man/examples.md",
    ]
```

## Documentation

```@contents
Pages = [
    "lib/public.md",
    "lib/internal.md",
    ]
```

## [Index](@id main-index)

```@index
Pages = [
    "lib/public.md",
    "lib/internal.md",
    ]
```

## References

+ Varin, C., N. Reid, and D. Firth (2011). An overview of composite likelihood methods. Statistica Sinica 21(1), 5–42.
