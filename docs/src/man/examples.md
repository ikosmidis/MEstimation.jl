# Examples

## Contents
```@contents
Pages = ["examples.md"]
Depth=3
```

## Ratio of two means

Consider a setting where independent pairs of random variables ``(X_1, Y_1), \ldots, (X_n, Y_n)`` are observed, and suppose that interest is in the ratio of the mean of ``Y_i`` to  the mean of ``X_i``, that is ``\theta = \mu_Y / \mu_X``, with
  ``\mu_X = E(X_i)`` and ``\mu_Y = E(Y_i) \ne 0`` ``(i = 1, \ldots, n)``.

Assuming that sampling is from an infinite population, one way of estimating ``\theta`` without any further assumptions about the joint distribution of ``(X_i, Y_i)`` is to set the unbiased estimating equation ``\sum_{i = 1}^n (Y_i - \theta X_i) = 0``. The resulting ``M``-estimator is then  ``\hat\theta = s_Y/s_X`` where ``s_X = \sum_{i = 1}^n X_i`` and ``s_Y = \sum_{i = 1}^n Y_i``. 

The estimator ``\hat\theta`` is generally biased, as can be shown, for example, by an application of the Jensen inequality assuming that ``X_i``is independent of ``Y_i``, and its bias can be reduced using the empirically adjusted estimating functions approach in Kosmidis & Lunardon (2020). 

This example illustrates how GEEBRA can be used to calculate the ``M``-estimator and its reduced-bias version.

```@repl 1
using GEEBRA, Random
```

Define a data type for ratio estimation problems
```@repl 1
struct ratio_data
    y::Vector
    x::Vector
end;
```

Write a function to compute the number of observations for objects of type `ratio_data`.
```@repl 1
function ratio_nobs(data::ratio_data)
    nx = length(data.x)
    ny = length(data.y)
    if (nx != ny) 
        error("length of x is not equal to the length of y")
    end
    nx
end;
```

Generate some data to test things out
```@repl 1
Random.seed!(123);
my_data = ratio_data(randn(10), rand(10));
ratio_nobs(my_data)
```

The estimating function for the ratio ``\theta`` is 

``\sum_{i = 1}^n (Y_i - \theta X_i)``

So, the contribution to the estimating function can be implemented as
```@repl 1
function ratio_ef(theta::Vector,
                  data::ratio_data,
                  i::Int64)
    data.y[i] .- theta * data.x[i]
end;
```

The `estimating_function_template` for the ratio estimation problem can now be set up using `ratio_nobs` and `ratio_ef`.
```@repl 1
    ratio_template = estimating_function_template(ratio_nobs, ratio_ef);
```

We are now ready use `ratio_template` and `my_data` to compute the ``M``-estimator of ``\theta`` by solving the esitmating equation ``\sum_{i = 1}^n (Y_i - \theta X_i) = 0``. The starting value for the nonlinear solver is set to `0.1`.
```@repl 1
result_m = fit(ratio_template, my_data, [0.1], false)
```
`fit` uses methods from the [**NLsolve**](https://github.com/JuliaNLSolvers/NLsolve.jl) package for solving the estimating equations. Arguments can be passed directly to `NLsolve.nlsolve` through [keyword arguments](https://docs.julialang.org/en/v1/manual/functions/#Keyword-Arguments-1) to the `fit` method. For example,
```@repl 1
result_m = fit(ratio_template, my_data, [0.1], false, show_trace = true)
```

Bias reduction in general ``M``-estimation can be achieved by solving the adjusted estimating equation ``\sum_{i = 1}^n (Y_i - \theta X_i) + A(\theta, Y, X) = 0``, where ``A(\theta)`` are empirical bias-reducing adjustments depending on the first and second derivatives of the estimating function contributions. **GEEBRA** can use `ratio_template` and automatic differentiation (see, [ForwardDiff](https://github.com/JuliaDiff/ForwardDiff.jl)) to construct ``A(\theta, Y, X)`` and, then, solve the bias-reducing adjusted estimating equations. All this is simply done by
```@repl 1
result_br = fit(ratio_template, my_data, [0.1], true) 
```

Kosmidis & Lunardon (2020) show that the reduced-bias estimator of $\theta$ is ``\tilde\theta = (s_Y + s_{XY}/s_{X})/(s_X + s_{XX}/s_{X})``. The code chunks below tests that this is indeed the result **GEEBRA** returns.
```@repl 1
sx = sum(my_data.x);
sxx = sum(my_data.x .* my_data.x);
sy = sum(my_data.y);
sxy = sum(my_data.x .* my_data.y);
isapprox(sy/sx, result_m.theta[1])
isapprox((sy + sxy/sx)/(sx + sxx/sx), result_br.theta[1])
```


## Logistic regression
### Using [`objective_function_template`](@ref)
Here, we use **GEEBRA**'s [`objective_function_template`](@ref) to estimate a logistic regression model using maximum likelihood and maximum penalized likelihood, with the empirical bias-reducing penalty in Kosmidis & Lunardon (2020).

```@repl 2
using GEEBRA
using Random
using Distributions
using Optim
```

A data type for logistic regression models (consisting of a response vector `y`, a model matrix `x`, and a vector of weights `m`) is
```@repl 2
struct logistic_data
    y::Vector
    x::Array{Float64}
    m::Vector
end
```

A function to compute the number of observations from `logistic_data` objects is
```@repl 2
function logistic_nobs(data::logistic_data)
    nx = size(data.x)[1]
    ny = length(data.y)
    nm = length(data.m)
    if (nx != ny) 
        error("number of rows in of x is not equal to the length of y")
    elseif (nx != nm)
        error("number of rows in of x is not equal to the length of m")
    elseif (ny != nm)
        error("length of y is not equal to the length of m")
    end
    nx
end
```

The logistic regression log-likelihood contribution at a parameter `theta` for the ``i``th observations of data `data` is
```@repl 2
function logistic_loglik(theta::Vector,
                         data::logistic_data,
                         i::Int64)
    eta = sum(data.x[i, :] .* theta)
    mu = exp.(eta)./(1 .+ exp.(eta))
    data.y[i] .* log.(mu) + (data.m[i] - data.y[i]) .* log.(1 .- mu)
end
```

Let's simulate some logistic regression data with $10$ covariates
```@repl 2
Random.seed!(123);
n = 100;
m = 1;
p = 10
x = Array{Float64}(undef, n, p);
x[:, 1] .= 1.0;
for j in 2:p
        x[:, j] .= rand(n);
end
true_betas = randn(p) * sqrt(p);
y = rand.(Binomial.(m, cdf.(Logistic(), x * true_betas)));
my_data = logistic_data(y, x, fill(m, n));
```
and set up an `objective_function_template` for logistic regression
```@repl 2
logistic_template = objective_function_template(logistic_nobs, logistic_loglik)
```

The maximum likelihood estimates starting at `true_betas` are
```@repl 2
o1_ml = fit(logistic_template, my_data, true_betas, false, method = NelderMead())
```
`fit` uses methods from the [**Optim**](https://github.com/JuliaNLSolvers/Optim.jl) package internally. Here, we used the `Optim.NelderMead` method. Alternative optimization methods and options can be supplied directly through the [keyword arguments](https://docs.julialang.org/en/v1/manual/functions/#Keyword-Arguments-1) `method` and `optim.Options`, respectively. For example,
```@repl 2
o2_ml = fit(logistic_template, my_data, true_betas, false, method = LBFGS(), optim_options = Optim.Options(g_abstol = 1e-05))
```

The reduced-bias estimates starting at the maximum likelihood ones are
```@repl 2
o1_br = fit(logistic_template, my_data, coef(o1_ml), true)
```

### Using [`estimating_function_template`](@ref)
The same results as above can be returned using an [`estimating_function_template`](@ref) for logistic regression. 

The contribution to the derivatives of the log-likelihood for logistic regression is
```@repl 2
function logistic_ef(theta::Vector,
                     data::logistic_data,
                     i::Int64)
    eta = sum(data.x[i, :] .* theta)
    mu = exp.(eta)./(1 .+ exp.(eta))
    data.x[i, :] * (data.y[i] - data.m[i] * mu)
end
```

Then, solving the bias-reducing adjusted estimating equations
```@repl 2
logistic_ef_template = estimating_function_template(logistic_nobs, logistic_ef);
e1_br = fit(logistic_ef_template, my_data, true_betas, true)
```
returns the reduced-bias estimates from maximum penalized likelihood:
```@repl 2
isapprox(coef(o1_br), coef(e1_br))
```

