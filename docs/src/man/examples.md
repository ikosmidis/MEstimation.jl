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

The estimator ``\hat\theta`` is generally biased, as can be shown, for example, by an application of the Jensen inequality assuming that ``X_i``is independent of ``Y_i``, and its bias can be reduced using the empirically adjusted estimating functions approach in [Kosmidis & Lunardon (2020)](http://arxiv.org/abs/2001.03786). 

This example illustrates how **MEstimation** can be used to calculate the ``M``-estimator and its reduced-bias version.

```@repl 1
using MEstimation, Random
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

We are now ready use `ratio_template` and `my_data` to compute the ``M``-estimator of ``\theta`` by solving the estimating equation ``\sum_{i = 1}^n (Y_i - \theta X_i) = 0``. The starting value for the nonlinear solver is set to `0.1`.
```@repl 1
result_m = fit(ratio_template, my_data, [0.1])
```
`fit` uses methods from the [**NLsolve**](https://github.com/JuliaNLSolvers/NLsolve.jl) package for solving the estimating equations. Arguments can be passed directly to `NLsolve.nlsolve` through [keyword arguments](https://docs.julialang.org/en/v1/manual/functions/#Keyword-Arguments-1) to the `fit` method. For example,
```@repl 1
result_m = fit(ratio_template, my_data, [0.1], show_trace = true)
```

Bias reduction in general ``M``-estimation can be achieved by solving the adjusted estimating equation ``\sum_{i = 1}^n (Y_i - \theta X_i) + A(\theta, Y, X) = 0``, where ``A(\theta)`` are empirical bias-reducing adjustments depending on the first and second derivatives of the estimating function contributions. **MEstimation** can use `ratio_template` and automatic differentiation (see, [ForwardDiff](https://github.com/JuliaDiff/ForwardDiff.jl)) to construct ``A(\theta, Y, X)`` and, then, solve the bias-reducing adjusted estimating equations. All this is simply done by
```@repl 1
result_br = fit(ratio_template, my_data, [0.1], estimation_method = "RBM") 
```
where `RBM` stands for reduced-bias `M`-estimation.

[Kosmidis & Lunardon (2020)](http://arxiv.org/abs/2001.03786) show that the reduced-bias estimator of ``\theta`` is ``\tilde\theta = (s_Y + s_{XY}/s_{X})/(s_X + s_{XX}/s_{X})``. The code chunks below tests that this is indeed the result **MEstimation** returns.
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
Here, we use **MEstimation**'s [`objective_function_template`](@ref) to estimate a logistic regression model using maximum likelihood and maximum penalized likelihood, with the empirical bias-reducing penalty in [Kosmidis & Lunardon (2020)](http://arxiv.org/abs/2001.03786).

```@repl 2
using MEstimation
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

Let's simulate some logistic regression data with ``10`` covariates
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
o1_ml = fit(logistic_template, my_data, true_betas, optim_method = NelderMead())
```
`fit` uses methods from the [**Optim**](https://github.com/JuliaNLSolvers/Optim.jl) package internally. Here, we used the `Optim.NelderMead` method. Alternative optimization methods and options can be supplied directly through the [keyword arguments](https://docs.julialang.org/en/v1/manual/functions/#Keyword-Arguments-1) `optim_method` and `optim_options`, respectively. For example,
```@repl 2
o2_ml = fit(logistic_template, my_data, true_betas, optim_method = LBFGS(), optim_options = Optim.Options(g_abstol = 1e-05))
```

The reduced-bias estimates starting at the maximum likelihood ones are
```@repl 2
o1_br = fit(logistic_template, my_data, coef(o1_ml), estimation_method = "RBM")
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
logistic_template_ef = estimating_function_template(logistic_nobs, logistic_ef);
e1_br = fit(logistic_template_ef, my_data, true_betas, estimation_method = "RBM")
```
returns the reduced-bias estimates from maximum penalized likelihood:
```@repl 2
isapprox(coef(o1_br), coef(e1_br))
```

### Bias-reduction methods
**MEstimation** currently implements 2 alternative bias reduction methods, called `implicit_trace` and `explicit_trace`. `implicit_trace` will adjust the estimating functions or penalize the objectives, as we have seen earlier. `explicit_trace`, on the other hand, will form an estimate of the bias of the ``M``-estimator and subtract that from the ``M``-estimates. The default method is `implicit_trace`.

For example, for logistic regression via estimating functions 
```@repl 2
e2_br = fit(logistic_template_ef, my_data, true_betas, estimation_method = "RBM", br_method = "explicit_trace")
```
which gives slightly different estimates that what are in the `implict_trace` fit in `e1_br`. 

The same can be done using objective functions, but numerical differentiation (using the [FiniteDiff](https://github.com/JuliaDiff/FiniteDiff.jl) package) is used to approximate the gradient of the bias-reducing penalty (i.e. ``A(\theta)``).
```@repl 2
o2_br = fit(logistic_template, my_data, true_betas, estimation_method = "RBM", br_method = "explicit_trace")
isapprox(coef(e2_br), coef(o2_br))
```

## Regularization
**MEstimation** allows to pass arbitrary regularizers to either the objective or the estimating functions. Below we illustrate that functionality for carrying out ridge logistic regression, and maximum penalized likelihood, with a Jeffreys-prior penalty.

### Ridge logistic regression
The `logistic_template` that we defined earlier can be used for doing L2-regularized logistic regression (aka ridge logistic regression); we only need to define a function that implements the L2 regularizer
```@repl 2
l2_penalty = (theta, data, λ) -> - λ * sum(theta.^2);
```

Then, the coefficient path can be computed as
```@repl 2
lambda = collect(0:0.5:10);
deviance = similar(lambda);
coefficients = Matrix{Float64}(undef, length(lambda), length(true_betas));
coefficients[1, :] = coef(o1_ml);
for j in 2:length(lambda)
    current_fit = fit(logistic_template, my_data, coefficients[j - 1, :],
                      regularizer = (theta, data) -> l2_penalty(theta, data, lambda[j]))
    deviance[j] = 2 * current_fit.results.minimum
    coefficients[j, :] = coef(current_fit)
end
```

The coefficients versus ``\lambda``, and the deviance values are then
```@repl 2
using Plots
plot(lambda, coefficients);
savefig("coef_path1.svg");
```
![](coef_path1.svg)

```@repl 2
plot(deviance, coefficients);
savefig("coef_path2.svg");
```
![](coef_path2.svg)

Another way to get the above is to define a new data type that has a filed for ``\lambda`` and then pass
```@repl 2
l2_penalty = (theta, data) -> - data.λ * sum(theta.^2)
```
to the `regularizer` argument when calling `fit`. Such a new data type, though, would require to redefine `logistic_loglik`, `logistic_nobs` and `logistic_template`.

### Jeffreys-prior penalty for bias reduction
[Firth (1993)](https://www.jstor.org/stable/2336755) showed that an alternative bias-reducing penalty for the logistic regression likelihood is the Jeffreys prior,. which can readily implemented and passed to `fit` through the `regularizer` interface that **MEstimation** provides. The logarithm of the Jeffreys prior for logistic regression is 
```@repl 2
using LinearAlgebra

function log_jeffreys_prior(theta, data)
    x = data.x
    probs = cdf.(Logistic(), x * theta)
    log(det((x .* (data.m .* probs .* (1 .- probs)))' * x)) / 2
end
```

Then, the reduced-bias estimates of [Firth (1993)](https://www.jstor.org/stable/2336755) are
```@repl 2
o_jeffreys = fit(logistic_template, my_data, true_betas, regularizer = log_jeffreys_prior)
```

Note here, that the `regularizer` is only used to get estimates. Then all model quantities are computed at those estimates, but based only on `logistic_loglik` (i.e. without adding the regularizer to it). [Kosmidis & Firth (2020)](http://arxiv.org/abs/1812.01938) provide a more specific procedure for computing the reduced-bias estimates from the penalization of the logistic regression likelihood by Jeffreys prior, which uses repeated maximum likelihood fits on adjusted binomial data. [Kosmidis & Firth (2020)](http://arxiv.org/abs/1812.01938) also show that, for logistic regression, the reduced-bias estimates from are always finite and shrink to zero relative to the maximum likelihood estimator.

Regularization is also available when fitting an `estimating_function_template`. For example, the gradient of the `log_jeffreys_prior` above is
```@repl 2
using ForwardDiff
log_jeffreys_prior_grad = (theta, data) -> ForwardDiff.gradient(pars -> log_jeffreys_prior(pars, data), theta)
```

Then the same fit as `o_jeffreys` can be obtained using estimating functions
```@repl 2
e_jeffreys = fit(logistic_template_ef, my_data, true_betas, regularizer = log_jeffreys_prior_grad)
```
Note here that the value of the estimating functions shown in the output is that of the gradient of the log-likelihood, i.e.
```@repl 2
logistic_loglik_grad = estimating_function(coef(e_jeffreys), my_data, logistic_template_ef)
```
instead of the regularized estimating functions, which, as expected, are very close to zero at the estimates
```@repl 2
logistic_loglik_grad .+ log_jeffreys_prior_grad(coef(e_jeffreys), my_data)
```


