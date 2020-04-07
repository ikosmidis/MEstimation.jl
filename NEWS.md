# MEstimation 0.1.0

## New functionality
+ New `concentrate` keyword argument in `fit` for `estimating_function_template`s, which allows adding bias-reducing adjustments only to a subset of estimating functions.
+ New `lower` and `upper` keyword arguments in `fit` for `objective_function_template`s, which allows estimation in constrained parameters spaces.
+ New `regularizer` keyword argument in `fit` to allows for user-supplier regularizer functions. 
+ New `slice` method for computing one-dimensional slices of objective and estimating functions. 
+ Keyword arguments can be passed directly to `Optim.optimize` (e.g. `autodiff = :forward`) through the `fit` interface for `objective_function_template`s.

## Bug fixes
+ `objective_function` and `estimating_function` are fully differentiable.

## Other improvements, updates and additions
+ The default output from `fit` now reports whether the optimization algorithm converged or not, and details on the (regularized) objective or estimating equations that are used.
+ Documentation written from scratch, and updated example in online documentation.
+ New tests.
+ Run time optimizations, mainly through using [`DiffResults`](https://github.com/JuliaDiff/DiffResults.jl), codebase refactoring, and explicit type specification.
+ Updates in compatibility with dependences.
