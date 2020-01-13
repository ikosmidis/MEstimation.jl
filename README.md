# GEEBRA.jl

**G**eneral **E**stimating **E**quations with or without **B**ias-**R**educing **A**djustments (pronounced `zee· bruh`)

[![Travis status](https://travis-ci.com/ikosmidis/GEEBRA.jl.svg?branch=master)](https://travis-ci.org/ikosmidis/GEEBRA.jl)
[![Coverage Status](https://img.shields.io/codecov/c/github/ikosmidis/GEEBRA.jl/master.svg)](https://codecov.io/github/ikosmidis/GEEBRA.jl?branch=master)
[![](https://img.shields.io/badge/docs-dev-red.svg)](https://ikosmidis.github.io/GEEBRA.jl/dev/)
[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://ikosmidis.github.io/GEEBRA.jl/stable/)
[![](https://img.shields.io/github/license/ikosmidis/GEEBRA.jl)](https://github.com/ikosmidis/GEEBRA.jl/blob/master/LICENSE.md)

## Package description

**GEEBRA** is a Julia package that implements M-estimation for
statistical models, either by solving estimating equations or by
maximizing inference objectives, like
[likelihoods](https://en.wikipedia.org/wiki/Likelihood_function) and
composite likelihoods (see, [Varin et al,
2011](http://www3.stat.sinica.edu.tw/statistica/oldpdf/A21n11.pdf),
for a review), using only user-specified templates of the estimating
function or the objective functions contributions.

A key feature is the use of only those templates and forward mode
automatic differentiation (as implemented in
[**ForwardDiff**](https://github.com/JuliaDiff/ForwardDiff.jl)) to
provide methods for **reduced-bias M-estimation**
(**RBM-estimation**). 

See the [documentation](https://ikosmidis.github.io/GEEBRA.jl/dev/)
for more information, and the
[examples](https://ikosmidis.github.io/GEEBRA.jl/dev/man/examples/)
for a showcase of the functionaly **GEEBRA** provides.

## Authors

| [**Ioannis Kosmidis**](http://www.ikosmidis.com) | **(author, maintainer)** |
--- | ---
| [**Nicola Lunardon**](https://www.unimib.it/nicola-lunardon) | **(author)** |

## References

+ Varin, C., N. Reid, and D. Firth (2011). An overview of composite likelihood methods. *Statistica Sinica 21*(1), 5-42. 
[PDF link](http://www3.stat.sinica.edu.tw/statistica/oldpdf/A21n11.pdf)
