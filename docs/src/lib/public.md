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
fit(template::objective_function_template, data::Any, theta::Vector; estimation_method::String = "M", br_method::String = "implicit_trace", optim_method = LBFGS(), optim_options = Optim.Options())
fit(template::estimating_function_template, data::Any, theta::Vector; estimation_method::String = "M", br_method::String = "implicit_trace", nlsolve_arguments...)
coef
vcov
stderror
coeftable
tic
aic
```
