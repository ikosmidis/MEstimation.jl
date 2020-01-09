# GEEBRA.jl

**GEEBRA** (General Estimating Equations with or without Bias-Reducing
Adjustments; pronounced `zee· bruh`) is a Julia package that provides
infrastructure to estimate statistical models by solving estimating
equations or by maximizing inference objectives, like the likelihood
functions, using user-specified templates. A key feature is the option
to adjust the estimating equation or penalize the objectives in order
to reduce the bias of the resulting estimators.

## M-estimation and bias reduction

**GEEBRA** has been designed so that the only requirements from the
user are to:
1. implement a [Julia composite type](https://docs.julialang.org/en/v1/manual/types/index.html) for the data;
2. implement a function for calculating the contribution to the estimating function or to the objective function from a single observation that has arguments the parameter vector, the data object, and the observation index;
3. implement a function for computing the number of observations from the data object;
4. specify a GEEBRA template (using `estimating_function_template` for estimating functions and `objective_function_template` for objective function) that has fields the functions for computing the contributions to the estimating functions or to the objective, and the number of observations.

**GEEBRA**, then, provides methods to estimate the unknown parameters by solving the estimating equations or maximizing the inference objectives. 

There is also the option to compute adjustments to the estimating functions or penalties to the objective to compute estimators with improved bias properties. 

The bias-reducing adjustments or penalties are formed internally from the supplied templates and data object, using forward mode automatic differentiation, as implemented in [**ForwardDiff**](https://github.com/JuliaDiff/ForwardDiff.jl). No further work or input is required by the user.

## Manual outline
```@contents
Pages = [
    "man/guide.md",
    "man/examples.md",
    ]
Depth = 1
```

## Library outline

```@contents
Pages = [
    "lib/public.md"
    ]
```

### [Index](@id main-index)

```@index
Pages = [
    "lib/public.md"
    ]
```