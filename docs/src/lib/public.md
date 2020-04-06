# Public documentation

## Contents

```@contents
Pages = ["public.md"]
```

## Index

```@index
Pages = ["public.md"]
```

## Public interface

```@docs
estimating_function_template
get_estimating_function
estimating_function
objective_function_template
objective_function
fit(template::objective_function_template, data::Any, theta::Vector{Float64}; estimation_method::String = "M", br_method::String = "implicit_trace", regularizer::Function = function regularizer(theta::Vector{Float64}, data::Any) Vector{Float64}() end, lower::Vector{Float64} = Vector{Float64}(), upper::Vector{Float64} = Vector{Float64}(), optim_method = LBFGS(), optim_options = Optim.Options(), optim_arguments...)
fit(template::estimating_function_template, data::Any, theta::Vector{Float64}; estimation_method::String = "M", br_method::String = "implicit_trace", concentrate::Vector{Int64} = Vector{Int64}(), regularizer::Function = function regularizer(theta::Vector{Float64}, data::Any) Vector{Float64}() end, nlsolve_arguments...)
coef
vcov
stderror
coeftable
tic
aic
slice
```
