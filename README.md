# MEstimation.jl

**Methods for M-estimation of statistical models**

[![Travis status](https://travis-ci.com/ikosmidis/MEstimation.jl.svg?branch=master)](https://travis-ci.org/ikosmidis/MEstimation.jl)
[![Coverage Status](https://img.shields.io/codecov/c/github/ikosmidis/MEstimation.jl/master.svg)](https://codecov.io/github/ikosmidis/MEstimation.jl?branch=master)
[![](https://img.shields.io/badge/docs-dev-red.svg)](https://ikosmidis.github.io/MEstimation.jl/dev/)
[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://ikosmidis.github.io/MEstimation.jl/stable/)
[![](https://img.shields.io/github/license/ikosmidis/MEstimation.jl)](https://github.com/ikosmidis/MEstimation.jl/blob/master/LICENSE.md)

## Package description

**MEstimation** is a Julia package that implements M-estimation for
statistical models (see, e.g. Stefanski and Boos, 2002, for an
accessible review) either by solving estimating equations or by
maximizing inference objectives, like
[likelihoods](https://en.wikipedia.org/wiki/Likelihood_function) and
composite likelihoods (see, [Varin et al,
2011](http://www3.stat.sinica.edu.tw/statistica/oldpdf/A21n11.pdf),
for a review), using user-specified templates of just
1. the estimating function or the objective functions contributions
2. a function to compute the number of independent contributions in a given data set

A key feature is the use of those templates along with forward mode
automatic differentiation (as implemented in
[**ForwardDiff**](https://github.com/JuliaDiff/ForwardDiff.jl)) to
provide methods for **reduced-bias M-estimation** (**RBM-estimation**;
see, [Kosmidis & Lunardon, 2020](http://arxiv.org/abs/2001.03786)).

See the [documentation](https://ikosmidis.github.io/MEstimation.jl/dev/)
for more information, and the
[examples](https://ikosmidis.github.io/MEstimation.jl/dev/man/examples/)
for a showcase of the functionality **MEstimation** provides.

See
[NEWS.md](https://github.com/ikosmidis/MEstimation.jl/blob/master/NEWS.md)
for changes, bug fixes and enhancements.

## Authors

| [**Ioannis Kosmidis**](http://www.ikosmidis.com) | **(author, maintainer)** |
--- | ---
| [**Nicola Lunardon**](https://www.unimib.it/nicola-lunardon) | **(author)** |

## References

+ Varin C, Reid N, and Firth D (2011). An overview of composite likelihood methods. *Statistica Sinica 21*(1), 5-42. [Link](http://www3.stat.sinica.edu.tw/statistica/oldpdf/A21n11.pdf)
+ Kosmidis I, Lunardon N (2020). Empirical bias-reducing adjustments to estimating functions. ArXiv:2001.03786. [Link](http://arxiv.org/abs/2001.03786)
+ Stefanski L A and Boos D D (2002). The calculus of M-estimation. *The American Statistician*(56), 29-38. [Link](https://www.jstor.org/stable/3087324)

