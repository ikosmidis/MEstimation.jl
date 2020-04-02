# GEEBRA 0.2.0

+ Added support for bias reduction of a subset of the parameters (only for `estimating_function_template`)
+ Added option for optimization in constrained parameters spaces (only for `objective_function_template`)
+ Added option to provide own regularizer function (either for objective or estimating functions)
+ Added slices for computing slices of objectives and estimating functions
+ The default output now reports whether the optimization algorithm converged or not
+ Documentation improvements and updates
+ Optimization improvements
+ Work to make code fully differentiable (estimating_function and objective_function)
+ Allowed to pass keyword arguments to Optim.optimize (e.g. autodiff = :forward)

# GEEBRA 0.1.0

First public release
