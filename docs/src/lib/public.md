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
fit(template::objective_function_template, data::Any, theta::Vector, br::Bool = false; method = LBFGS(), optim_Options = Optim.Options())
fit(template::estimating_function_template, data::Any, theta::Vector, br::Bool = false; nlsolve_arguments...)
coef
vcov
stderror
coeftable
tic
aic
```
