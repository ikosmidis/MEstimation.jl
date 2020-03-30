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
fit(template::estimating_function_template, data::Any, theta::Vector{Float64}; estimation_method::String = "M", br_method::String = "implicit_trace", concentrate::Vector{Int64} = Vector{Int64}(), regularizer::Function = function regularizer(theta::Vector{Float64}, data::Any) Vector{Float64}() end, nlsolve_arguments...)
fit(template::objective_function_template, data::Any, theta::Vector{Float64}; lower::Vector{Float64} = Vector{Float64}(), upper::Vector{Float64} = Vector{Float64}(), estimation_method::String = "M", br_method::String = "implicit_trace", optim_method = LBFGS(), optim_options = Optim.Options(), regularizer::Function = function regularizer(theta::Vector{Float64}, data::Any) Vector{Float64}() end)
coef
vcov
stderror
coeftable
tic
aic
```
