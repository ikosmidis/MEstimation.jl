# GEEBRA.jl

**G**eneral **E**stimating **E**quations with or without **B**ias-**R**educing **A**djustments (pronounced `zee· bruh`)

[![Travis status](https://travis-ci.com/ikosmidis/GEEBRA.jl.svg?branch=master)](https://travis-ci.org/ikosmidis/GEEBRA.jl)
[![Coverage Status](https://img.shields.io/codecov/c/github/ikosmidis/GEEBRA.jl/master.svg)](https://codecov.io/github/ikosmidis/GEEBRA.jl?branch=master)
[![](https://img.shields.io/badge/docs-dev-red.svg)](https://ikosmidis.github.io/GEEBRA.jl/dev/)
[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://ikosmidis.github.io/GEEBRA.jl/stable/)
[![](https://img.shields.io/github/license/ikosmidis/GEEBRA.jl)](https://github.com/ikosmidis/GEEBRA.jl/blob/master/LICENSE.md)

## Package description

**GEEBRA** is a Julia package that provides infrastructure to estimate
statistical models by solving estimating equations or by maximizing
inference objectives, like
[likelihood](https://en.wikipedia.org/wiki/Likelihood_function) and
composite likelihood functions (see, [Varin et al, 2011](http://www3.stat.sinica.edu.tw/statistica/oldpdf/A21n11.pdf), for a review),
using user-specified templates.

A key feature is the option to adjust the estimating equation or
penalize the objectives in order to reduce the bias of the resulting
estimators.

**See the [GEEBRA documentation](https://ikosmidis.github.io/GEEBRA.jl/dev/) for more
information and
[examples](https://ikosmidis.github.io/GEEBRA.jl/dev/man/examples/)**

## Authors

| [Ioannis Kosmidis](http://www.ikosmidis.com) | **(author, maintainer)** |
--- | ---
| [Nicola Lunardon](https://www.unimib.it/nicola-lunardon) | **(author)** |

## References

+ Varin, C., N. Reid, and D. Firth (2011). An overview of composite likelihood methods. *Statistica Sinica 21*(1), 5-42. 
[PDF link](http://www3.stat.sinica.edu.tw/statistica/oldpdf/A21n11.pdf)
